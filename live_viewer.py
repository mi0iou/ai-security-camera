#!/usr/bin/env python3
"""
Live Detection Viewer
Shows camera feed with bounding boxes and labels
High-resolution capture for accurate detection, scaled display for performance
"""

import cv2
import numpy as np
import yaml
import time
from picamera2 import Picamera2
from hailo_detector import HailoDetector


class LiveViewer:
    def __init__(self):
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Capture resolution (what detector processes - full res for accuracy)
        self.capture_width = 1920
        self.capture_height = 1080
        
        # Display resolution (what you see on screen - scaled for performance)
        self.display_width = 960
        self.display_height = 540
        
        # Pre-calculate scaling factors for display
        self.scale_x = self.display_width / self.capture_width
        self.scale_y = self.display_height / self.capture_height
        
        # Initialize camera at full resolution
        print(f"Initializing camera at {self.capture_width}x{self.capture_height}...")
        self.cam = Picamera2(0)
        config = self.cam.create_preview_configuration(
            main={"size": (self.capture_width, self.capture_height), "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(2)
        
        # Initialize detector
        print("Loading Hailo detector...")
        self.detector = HailoDetector(
            hef_path=self.config['detection']['hailo_model_path'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            classes_to_detect=self.config['detection']['classes_to_detect']
        )
        
        print(f"Ready! Capture: {self.capture_width}x{self.capture_height}, Display: {self.display_width}x{self.display_height}")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Colors for different classes (BGR format for cv2)
        self.colors = {
            0: (255, 0, 0),      # person - blue
            1: (255, 128, 0),    # bicycle - light blue
            2: (0, 255, 0),      # car - green
            3: (0, 255, 255),    # motorcycle - yellow
            5: (0, 165, 255),    # bus - orange
            7: (0, 0, 255),      # truck - red
        }
        self.default_color = (255, 255, 0)  # cyan for others
        
        # FPS calculation
        self.fps_history = []
        self.fps = 0
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame
        
        Args:
            frame: Display-resolution frame (already scaled)
            detections: List of detections with bbox [x1,y1,x2,y2] in capture resolution
        """
        for det in detections:
            # Get bbox coordinates (in capture resolution)
            x1, y1, x2, y2 = det['bbox']
            
            # Scale bounding box from capture resolution to display resolution
            x1 = int(x1 * self.scale_x)
            y1 = int(y1 * self.scale_y)
            x2 = int(x2 * self.scale_x)
            y2 = int(y2 * self.scale_y)
            
            # Clamp to display bounds
            x1 = max(0, min(x1, self.display_width - 1))
            y1 = max(0, min(y1, self.display_height - 1))
            x2 = max(0, min(x2, self.display_width - 1))
            y2 = max(0, min(y2, self.display_height - 1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            class_id = det['class']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for this class
            color = self.colors.get(class_id, self.default_color)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label position and size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Ensure label stays within frame
            label_y1 = y1 - label_h - 10
            label_y2 = y1
            
            # If label would go above frame, put it below the box
            if label_y1 < 0:
                label_y1 = y2
                label_y2 = y2 + label_h + 10
            
            # Draw label background
            cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 4, label_y2), color, -1)
            
            # Draw label text
            text_y = label_y2 - 5 if label_y1 < y1 else label_y1 + label_h + 3
            cv2.putText(frame, label, (x1 + 2, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return frame
    
    def draw_info(self, frame, detections, fps, inference_time):
        """Draw info overlay"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (220, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Inference time
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Resolution info
        cv2.putText(frame, f"Capture: {self.capture_width}x{self.capture_height}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status bar at bottom
        cv2.rectangle(frame, (0, height - 35), (width, height), (40, 40, 40), -1)
        cv2.putText(frame, "LIVE - Hailo-8L Accelerated", (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Q: Quit | S: Screenshot", (width - 180, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        screenshot_count = 0
        
        while True:
            loop_start = time.time()
            
            # Capture frame at full resolution
            frame_full = self.cam.capture_array()
            
            # Run detection on full resolution frame
            inference_start = time.time()
            detections = self.detector.detect(frame_full)
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Scale frame down for display
            frame_display = cv2.resize(frame_full, (self.display_width, self.display_height))
            
            # Convert RGB to BGR for cv2 display
            frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            
            # Draw detections (scaling bbox coords from capture to display resolution)
            frame_bgr = self.draw_detections(frame_bgr, detections)
            
            # Calculate FPS
            elapsed = time.time() - loop_start
            self.fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.fps = sum(self.fps_history) / len(self.fps_history)
            
            # Draw info overlay
            frame_bgr = self.draw_info(frame_bgr, detections, self.fps, inference_time)
            
            # Display
            cv2.imshow('Live Detection Viewer', frame_bgr)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                # Save both full-res and display versions
                filename_display = f"screenshot_{screenshot_count}_display.jpg"
                filename_full = f"screenshot_{screenshot_count}_full.jpg"
                cv2.imwrite(filename_display, frame_bgr)
                cv2.imwrite(filename_full, cv2.cvtColor(frame_full, cv2.COLOR_RGB2BGR))
                print(f"Screenshots saved: {filename_display}, {filename_full}")
        
        # Cleanup
        self.cam.stop()
        cv2.destroyAllWindows()
        print("Viewer closed.")


if __name__ == "__main__":
    try:
        viewer = LiveViewer()
        viewer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
