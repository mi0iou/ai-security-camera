#!/usr/bin/env python3
"""
Hailo-8L YOLO Detection Module
Compatible with HailoRT 4.23
Handles NMS post-processed YOLOv8 output with proper coordinate handling

IMPORTANT: Hailo yolov8_nms_postprocess outputs coordinates as [y1, x1, y2, x2]
normalized 0-1 relative to the 640x640 letterboxed image, NOT [x1, y1, x2, y2]!
"""

import numpy as np
import cv2
import logging
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams


class HailoDetector:
    def __init__(self, hef_path, confidence_threshold=0.5, classes_to_detect=None):
        self.logger = logging.getLogger('HailoDetector')
        self.confidence_threshold = confidence_threshold
        self.classes_to_detect = classes_to_detect
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.logger.info(f"Loading Hailo model from {hef_path}")
        
        # Load HEF
        self.hef = HEF(hef_path)
        
        # Create device
        params = VDevice.create_params()
        self.device = VDevice(params)
        
        # Configure network group
        network_groups = self.device.configure(self.hef)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()
        
        # Get input/output info
        input_vstream_infos = self.network_group.get_input_vstream_infos()
        output_vstream_infos = self.network_group.get_output_vstream_infos()
        
        # Get input shape and name from VStreamInfo
        input_info = input_vstream_infos[0]
        self.input_name = input_info.name
        self.input_height = input_info.shape[0]
        self.input_width = input_info.shape[1]
        self.input_channels = input_info.shape[2]
        
        # Create VStream parameters
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False
        )
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False
        )
        
        self.logger.info(f"Hailo model loaded. Input: {self.input_name}, Shape: {self.input_height}x{self.input_width}x{self.input_channels}")
    
    def letterbox(self, frame, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resize and pad image while maintaining aspect ratio.
        
        Args:
            frame: Input image (H, W, C)
            new_shape: Target size (height, width)
            color: Padding color (R, G, B)
        
        Returns:
            letterboxed: Padded image
            scale: Scale factor used
            pad: Padding (pad_w, pad_h) added
        """
        shape = frame.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute new unpadded dimensions
        new_w, new_h = int(shape[1] * r), int(shape[0] * r)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Compute padding
        pad_w = (new_shape[1] - new_w) // 2
        pad_h = (new_shape[0] - new_h) // 2
        
        # Create padded image
        letterboxed = np.full((new_shape[0], new_shape[1], 3), color[0], dtype=np.uint8)
        letterboxed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return letterboxed, r, (pad_w, pad_h)
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for Hailo with letterboxing to maintain aspect ratio.
        
        Returns:
            preprocessed: Letterboxed frame ready for inference
            original_shape: Original frame shape (height, width)
            scale: Scale factor used in letterboxing
            pad: Padding (pad_w, pad_h) added
        """
        original_height, original_width = frame.shape[:2]
        
        # Letterbox to model input size maintaining aspect ratio
        letterboxed, scale, pad = self.letterbox(
            frame, 
            new_shape=(self.input_height, self.input_width),
            color=(114, 114, 114)  # YOLO default gray
        )
        
        # Ensure correct data type (uint8)
        if letterboxed.dtype != np.uint8:
            letterboxed = letterboxed.astype(np.uint8)
        
        return letterboxed, (original_height, original_width), scale, pad
    
    def scale_coords(self, raw_coords, original_shape, scale, pad):
        """
        Scale coordinates from Hailo output to original image space.
        
        IMPORTANT: Hailo NMS outputs [y1, x1, y2, x2] NOT [x1, y1, x2, y2]!
        
        Args:
            raw_coords: [v0, v1, v2, v3] = [y1, x1, y2, x2] normalized 0-1 in letterboxed space
            original_shape: (original_height, original_width)
            scale: Scale factor used in letterboxing
            pad: (pad_w, pad_h) padding added during letterboxing
        
        Returns:
            Scaled coordinates [x1, y1, x2, y2] in original image pixel space
        """
        original_height, original_width = original_shape
        pad_w, pad_h = pad
        
        # Hailo outputs [y1, x1, y2, x2] - extract correctly
        y1_norm = raw_coords[0]
        x1_norm = raw_coords[1]
        y2_norm = raw_coords[2]
        x2_norm = raw_coords[3]
        
        # Convert from normalized (0-1) to pixel coordinates in 640x640 space
        x1_letterbox = x1_norm * self.input_width
        y1_letterbox = y1_norm * self.input_height
        x2_letterbox = x2_norm * self.input_width
        y2_letterbox = y2_norm * self.input_height
        
        # Subtract letterbox padding
        x1_scaled = x1_letterbox - pad_w
        y1_scaled = y1_letterbox - pad_h
        x2_scaled = x2_letterbox - pad_w
        y2_scaled = y2_letterbox - pad_h
        
        # Scale back to original image size
        x1 = x1_scaled / scale
        y1 = y1_scaled / scale
        x2 = x2_scaled / scale
        y2 = y2_scaled / scale
        
        # Clamp to original image bounds
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    
    def postprocess_detections(self, raw_outputs, original_shape, scale, pad):
        """
        Postprocess Hailo YOLOv8 NMS output.
        
        Args:
            raw_outputs: Raw model outputs
            original_shape: (original_height, original_width)
            scale: Scale factor from letterboxing
            pad: (pad_w, pad_h) from letterboxing
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for output_name, output_data in raw_outputs.items():
            try:
                # Handle list output - each element is detections for one class
                if isinstance(output_data, (list, tuple)):
                    # First level: remove batch dimension if present
                    if len(output_data) == 1:
                        output_data = output_data[0]
                    
                    num_classes = len(output_data)
                    
                    # Iterate through each class
                    for class_id in range(num_classes):
                        # Filter by classes we want
                        if self.classes_to_detect and class_id not in self.classes_to_detect:
                            continue
                        
                        # Get detections for this class
                        class_detections = output_data[class_id]
                        
                        # Convert to numpy array
                        if not isinstance(class_detections, np.ndarray):
                            class_detections = np.array(class_detections)
                        
                        # Skip if empty
                        if class_detections.size == 0 or len(class_detections.shape) == 0:
                            continue
                        
                        # Ensure 2D array [num_detections, 5]
                        if len(class_detections.shape) == 1:
                            class_detections = class_detections.reshape(1, -1)
                        
                        # Process each detection for this class
                        for detection in class_detections:
                            if len(detection) < 5:
                                continue
                            
                            confidence = float(detection[4])
                            
                            # Skip low confidence
                            if confidence < self.confidence_threshold:
                                continue
                            
                            # Raw coordinates from Hailo: [y1, x1, y2, x2, conf]
                            raw_coords = [
                                float(detection[0]),  # y1
                                float(detection[1]),  # x1
                                float(detection[2]),  # y2
                                float(detection[3])   # x2
                            ]
                            
                            # Scale coordinates back to original image space
                            # Returns [x1, y1, x2, y2] in standard format
                            bbox = self.scale_coords(raw_coords, original_shape, scale, pad)
                            
                            # Skip invalid boxes
                            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                                continue
                            
                            detections.append({
                                'class': int(class_id),
                                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                                'confidence': confidence,
                                'bbox': bbox  # [x1, y1, x2, y2] in original image coords
                            })
                
                else:
                    self.logger.debug(f"Unexpected output type: {type(output_data)}")
            
            except Exception as e:
                self.logger.error(f"Error processing output '{output_name}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return detections
    
    def detect(self, frame):
        """
        Run detection on frame.
        
        Args:
            frame: Input frame (RGB, any size)
        
        Returns:
            List of detection dictionaries with bbox [x1,y1,x2,y2] in original frame coordinates
        """
        try:
            # Preprocess with letterboxing
            input_data, original_shape, scale, pad = self.preprocess_frame(frame)
            
            # Prepare input (add batch dimension)
            input_array = np.expand_dims(input_data, axis=0)
            
            # Create input dictionary
            input_dict = {self.input_name: input_array}
            
            # Activate network group and run inference
            with self.network_group.activate(self.network_group_params):
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                    raw_outputs = infer_pipeline.infer(input_dict)
            
            # Postprocess with proper coordinate scaling
            detections = self.postprocess_detections(raw_outputs, original_shape, scale, pad)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Hailo inference error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'network_group'):
                self.network_group = None
            if hasattr(self, 'device'):
                self.device = None
        except:
            pass
