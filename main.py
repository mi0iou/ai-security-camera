#!/usr/bin/env python3
"""
AI Security Camera System
Dual camera setup with YOLO + ANPR on Raspberry Pi 5 + Hailo-8L
Now with frame sharing for web dashboard preview
"""

import cv2
import numpy as np
import threading
import queue
import time
import sqlite3
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from picamera2 import Picamera2
from ultralytics import YOLO
import requests
import base64

from anpr_module import ANPRProcessor
from database_manager import DatabaseManager
from alert_manager import AlertManager
from frame_buffer import frame_buffer  # Import shared frame buffer


class SecurityCamera:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.logger.info("Initializing Security Camera System...")
        
        self.db = DatabaseManager(self.config['database']['path'])
        self.alert_manager = AlertManager(self.config['alerts'], self.db)
        self.anpr = ANPRProcessor(self.config['anpr'])
        
        self.detection_cam = None
        self.anpr_cam = None
        self.setup_cameras()
        
        self.setup_yolo()
        
        self.detection_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.anpr_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        
        self.running = False
        self.threads = []
        
        # FPS tracking for dashboard
        self._fps_history = []
        self._last_inference_ms = 0
        
        # Person alert cooldown tracking
        self._last_person_alert = None
        self._person_alert_cooldown = self.config['alerts'].get('cooldown_seconds', 60)
        
        # Per-class detection logging cooldown
        # Only log to database once per cooldown period per class
        # Detections still appear on the live feed - just not spammed into the DB
        self._last_class_log = {}  # {class_id: datetime}
        self._detection_log_cooldown = self.config['performance'].get('detection_log_cooldown', 30)
        
        # ANPR trigger cooldown - don't keep capturing the same vehicle
        self._last_anpr_trigger = {}  # {class_id: datetime}
        self._anpr_trigger_cooldown = self.config['performance'].get('detection_log_cooldown', 30)
        
        self.logger.info(f"Detection log cooldown: {self._detection_log_cooldown}s per class")
        self.logger.info(f"ANPR trigger cooldown: {self._anpr_trigger_cooldown}s per vehicle class")
        self.logger.info("System initialized successfully")
    
    def setup_logging(self):
        log_config = self.config['logging']
        log_path = Path(log_config['file'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler() if log_config['console'] else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger('SecurityCamera')
    
    def setup_cameras(self):
        try:
            self.logger.info("Initializing detection camera (IMX296)...")
            self.detection_cam = Picamera2(self.config['cameras']['detection']['index'])
            det_config = self.detection_cam.create_preview_configuration(
                main={"size": tuple(self.config['cameras']['detection']['resolution']), 
                      "format": "RGB888"}
            )
            self.detection_cam.configure(det_config)
            self.detection_cam.start()
            time.sleep(2)
            
            self.logger.info("Initializing ANPR camera (IMX477)...")
            self.anpr_cam = Picamera2(self.config['cameras']['anpr']['index'])
            anpr_config = self.anpr_cam.create_preview_configuration(
                main={"size": tuple(self.config['cameras']['anpr']['resolution']), 
                      "format": "RGB888"}
            )
            self.anpr_cam.configure(anpr_config)
            self.anpr_cam.start()
            time.sleep(2)
            
            self.logger.info("Both cameras initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def setup_yolo(self):
        try:
            if self.config['detection']['use_hailo']:
                self.logger.info("Loading YOLO model with Hailo acceleration...")
                from hailo_detector import HailoDetector
                
                self.model = HailoDetector(
                    hef_path=self.config['detection']['hailo_model_path'],
                    confidence_threshold=self.config['detection']['confidence_threshold'],
                    classes_to_detect=self.config['detection']['classes_to_detect']
                )
                self.use_hailo = True
                self.logger.info("Hailo YOLO model loaded successfully")
            else:
                self.logger.info("Loading standard YOLO model...")
                self.model = YOLO(self.config['detection']['yolo_model'])
                self.use_hailo = False
                self.logger.info("Standard YOLO model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"YOLO initialization failed: {e}")
            self.logger.warning("Falling back to standard YOLO")
            self.model = YOLO(self.config['detection']['yolo_model'])
            self.use_hailo = False
    
    def _update_fps(self, elapsed):
        """Update FPS tracking"""
        if elapsed > 0:
            self._fps_history.append(1.0 / elapsed)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
    
    def _get_fps(self):
        """Get current average FPS"""
        if self._fps_history:
            return sum(self._fps_history) / len(self._fps_history)
        return 0
    
    def _should_send_person_alert(self):
        """Check if enough time has passed since last person alert"""
        if self._last_person_alert is None:
            return True
        
        elapsed = (datetime.now() - self._last_person_alert).total_seconds()
        return elapsed >= self._person_alert_cooldown
    
    def _should_log_detection(self, class_id):
        """
        Check if this class should be logged to the database.
        Returns True if cooldown has passed since last log for this class.
        This prevents the same object being counted thousands of times
        while it sits in frame.
        """
        now = datetime.now()
        
        if class_id not in self._last_class_log:
            self._last_class_log[class_id] = now
            return True
        
        elapsed = (now - self._last_class_log[class_id]).total_seconds()
        if elapsed >= self._detection_log_cooldown:
            self._last_class_log[class_id] = now
            return True
        
        return False
    
    def _log_detections_by_class(self, detections, frame):
        """
        Log detections to the database, respecting per-class cooldown.
        When cooldown expires for a class, logs ALL instances of that class
        visible in the current frame (e.g. 2 chairs = 2 events).
        
        Returns:
            person_count: Number of people logged this frame
            person_image_path: Path to saved person image (if any)
        """
        person_count = 0
        person_image_path = None
        
        if not detections:
            return person_count, person_image_path
        
        # Count how many of each class are in this frame
        class_counts = Counter(det['class'] for det in detections)
        
        # For each class present, check cooldown and log correct count
        for cls_id, count in class_counts.items():
            if self._should_log_detection(cls_id):
                # Get all detections of this class
                class_dets = [d for d in detections if d['class'] == cls_id]
                class_name = class_dets[0]['class_name']
                best_conf = max(d['confidence'] for d in class_dets)
                
                # Save one image for this class (if enabled)
                image_path = None
                if self.config['storage']['save_detection_frames']:
                    image_path = self.save_frame(frame, f"{class_name}_{cls_id}")
                
                self.logger.info(f"{class_name} x{count} detected (best confidence {best_conf:.2f})")
                
                # Log one event per instance so counts are accurate
                for det in class_dets:
                    self.db.log_event(
                        timestamp=datetime.now(),
                        event_type=class_name,
                        plate_number=None,
                        confidence=det['confidence'],
                        image_path=image_path
                    )
                
                # Track person detections for alerts
                if cls_id == 0:  # person
                    person_count = count
                    person_image_path = image_path
        
        return person_count, person_image_path
    
    def _trigger_anpr_for_vehicles(self, detections, frame):
        """Queue vehicle detections for ANPR processing (with cooldown)"""
        if not self.config['anpr']['enabled']:
            return
        
        now = datetime.now()
        
        for det in detections:
            if det['class'] in [2, 5, 7]:  # car, bus, truck
                cls_id = det['class']
                
                # Check cooldown for this vehicle class
                if cls_id in self._last_anpr_trigger:
                    elapsed = (now - self._last_anpr_trigger[cls_id]).total_seconds()
                    if elapsed < self._anpr_trigger_cooldown:
                        continue
                
                self._last_anpr_trigger[cls_id] = now
                
                try:
                    self.anpr_queue.put_nowait({
                        'timestamp': now,
                        'detection_frame': frame.copy(),
                        'bbox': det['bbox'],
                        'class': det['class'],
                        'confidence': det['confidence']
                    })
                except queue.Full:
                    self.logger.warning("ANPR queue full, skipping frame")
                    break
    
    def detection_loop(self):
        """Main detection loop on IMX296"""
        self.logger.info("Detection loop started")
        interval = self.config['performance']['detection_interval']
        classes_to_detect = self.config['detection']['classes_to_detect']
        confidence = self.config['detection']['confidence_threshold']
        
        # Frame publishing rate limiter (don't publish every frame)
        last_publish_time = 0
        publish_interval = 0.1  # Publish to dashboard at max 10 FPS
        
        while self.running:
            try:
                loop_start = time.time()
                
                frame = self.detection_cam.capture_array()
                
                inference_start = time.time()
                
                if self.use_hailo:
                    detections = self.model.detect(frame)
                    
                    self._last_inference_ms = (time.time() - inference_start) * 1000
                    
                    # Log detections to database (with per-class cooldown and correct counts)
                    person_count, person_image_path = self._log_detections_by_class(detections, frame)
                    
                    # Trigger ANPR for any vehicles detected
                    self._trigger_anpr_for_vehicles(detections, frame)
                    
                    # Send person alert if enabled and persons were logged
                    if person_count > 0 and self.config['alerts']['alert_on'].get('person_detected', False):
                        if self._should_send_person_alert():
                            self.logger.info(f"Sending person alert (count: {person_count})")
                            self.alert_manager.send_person_alert(
                                count=person_count,
                                image_path=person_image_path
                            )
                            self._last_person_alert = datetime.now()
                        else:
                            self.logger.debug("Person alert skipped (cooldown active)")
                    
                    # Publish frame to dashboard buffer (rate limited)
                    # This still shows ALL detections on the live feed
                    current_time = time.time()
                    if current_time - last_publish_time >= publish_interval:
                        stats = {
                            'fps': self._get_fps(),
                            'inference_ms': self._last_inference_ms,
                            'detection_count': len(detections)
                        }
                        frame_buffer.publish_frame(frame, detections, stats)
                        last_publish_time = current_time
                
                else:
                    results = self.model(frame, conf=confidence, classes=classes_to_detect, verbose=False)
                    
                    self._last_inference_ms = (time.time() - inference_start) * 1000
                    
                    # Build detection list for frame buffer (always, for live view)
                    detections_list = []
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            class_names = result.names
                            class_name = class_names[cls]
                            
                            detections_list.append({
                                'class': cls,
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            })
                    
                    # Log detections to database (with per-class cooldown and correct counts)
                    person_count, person_image_path = self._log_detections_by_class(detections_list, frame)
                    
                    # Trigger ANPR for any vehicles detected
                    self._trigger_anpr_for_vehicles(detections_list, frame)
                    
                    # Send person alert if enabled and persons were logged
                    if person_count > 0 and self.config['alerts']['alert_on'].get('person_detected', False):
                        if self._should_send_person_alert():
                            self.logger.info(f"Sending person alert (count: {person_count})")
                            self.alert_manager.send_person_alert(
                                count=person_count,
                                image_path=person_image_path
                            )
                            self._last_person_alert = datetime.now()
                        else:
                            self.logger.debug("Person alert skipped (cooldown active)")
                    
                    # Publish frame to dashboard buffer (rate limited)
                    current_time = time.time()
                    if current_time - last_publish_time >= publish_interval:
                        stats = {
                            'fps': self._get_fps(),
                            'inference_ms': self._last_inference_ms,
                            'detection_count': len(detections_list)
                        }
                        frame_buffer.publish_frame(frame, detections_list, stats)
                        last_publish_time = current_time
                
                # Update FPS tracking
                elapsed = time.time() - loop_start
                self._update_fps(elapsed)
                
                # Sleep if needed to maintain interval
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def _map_bbox_to_anpr(self, bbox):
        """
        Map a bounding box from the detection camera (IMX296 + 6mm)
        to the ANPR camera (IMX477 + 16mm) coordinate space.
        
        Uses the angular FOV ratio between the two lens/sensor combinations.
        Both cameras should be mounted close together pointing the same direction.
        
        IMX296 + 6mm:  ~55° horizontal FOV, 1920x1080
        IMX477 + 16mm: ~22° horizontal FOV, 4056x3040
        
        Returns mapped [x1, y1, x2, y2] in ANPR camera coordinates, or None if
        the detection falls outside the ANPR camera's narrower FOV.
        """
        det_w = self.config['cameras']['detection']['resolution'][0]  # 1920
        det_h = self.config['cameras']['detection']['resolution'][1]  # 1080
        anpr_w = self.config['cameras']['anpr']['resolution'][0]      # 4056
        anpr_h = self.config['cameras']['anpr']['resolution'][1]      # 3040
        
        # Pixels per degree for each camera (approximate)
        det_fov_h = 55.0   # degrees, 6mm lens on IMX296
        anpr_fov_h = 22.0  # degrees, 16mm lens on IMX477
        
        det_ppd = det_w / det_fov_h      # ~34.9 px/degree
        anpr_ppd = anpr_w / anpr_fov_h   # ~184.4 px/degree
        
        # Map each corner: detection pixel -> angle from centre -> ANPR pixel
        det_cx, det_cy = det_w / 2, det_h / 2
        anpr_cx, anpr_cy = anpr_w / 2, anpr_h / 2
        
        x1, y1, x2, y2 = bbox
        
        # Convert to angles from centre (degrees)
        ang_x1 = (x1 - det_cx) / det_ppd
        ang_y1 = (y1 - det_cy) / det_ppd
        ang_x2 = (x2 - det_cx) / det_ppd
        ang_y2 = (y2 - det_cy) / det_ppd
        
        # Convert angles to ANPR pixels
        ax1 = int(ang_x1 * anpr_ppd + anpr_cx)
        ay1 = int(ang_y1 * anpr_ppd + anpr_cy)
        ax2 = int(ang_x2 * anpr_ppd + anpr_cx)
        ay2 = int(ang_y2 * anpr_ppd + anpr_cy)
        
        # Add generous padding (30%) since alignment won't be perfect
        pad_x = int((ax2 - ax1) * 0.3)
        pad_y = int((ay2 - ay1) * 0.3)
        ax1 = max(0, ax1 - pad_x)
        ay1 = max(0, ay1 - pad_y)
        ax2 = min(anpr_w, ax2 + pad_x)
        ay2 = min(anpr_h, ay2 + pad_y)
        
        # Check the mapped region is within the ANPR frame
        if ax2 <= 0 or ay2 <= 0 or ax1 >= anpr_w or ay1 >= anpr_h:
            self.logger.debug("Vehicle bbox falls outside ANPR camera FOV")
            return None
        
        return [ax1, ay1, ax2, ay2]
    
    def anpr_loop(self):
        """ANPR processing loop on IMX477"""
        self.logger.info("ANPR loop started")
        
        while self.running:
            try:
                detection_data = self.anpr_queue.get(timeout=1)
                
                anpr_frame = self.anpr_cam.capture_array()
                
                # Map the vehicle bbox from detection camera to ANPR camera coordinates
                mapped_bbox = self._map_bbox_to_anpr(detection_data['bbox'])
                
                plate_result = self.anpr.read_plate(anpr_frame, vehicle_bbox=mapped_bbox)
                
                if plate_result and plate_result['confidence'] >= self.config['anpr']['min_confidence']:
                    plate_number = plate_result['plate']
                    confidence = plate_result['confidence']
                    
                    self.logger.info(f"Plate detected: {plate_number} (confidence: {confidence:.2f})")
                    
                    if self.config['storage']['save_anpr_frames']:
                        image_path = self.save_frame(anpr_frame, f"plate_{plate_number}")
                    else:
                        image_path = None
                    
                    plate_info = self.db.check_plate(plate_number)
                    
                    event_id = self.db.log_event(
                        timestamp=detection_data['timestamp'],
                        event_type='vehicle',
                        plate_number=plate_number,
                        confidence=confidence,
                        image_path=image_path
                    )
                    
                    if plate_info:
                        alert_type = plate_info['alert_type']
                        if alert_type == 'blacklist':
                            self.alert_manager.send_alert(
                                title="\u26a0\ufe0f BLACKLISTED VEHICLE",
                                message=f"Plate: {plate_number}\nOwner: {plate_info['owner_name']}",
                                priority=5,
                                image_path=image_path
                            )
                        elif alert_type == 'known' and self.config['alerts']['alert_on']['known_plate']:
                            self.alert_manager.send_alert(
                                title="Known Vehicle",
                                message=f"Plate: {plate_number}\nOwner: {plate_info['owner_name']}",
                                priority=3,
                                image_path=image_path
                            )
                        
                        self.db.update_last_seen(plate_number)
                    else:
                        if self.config['alerts']['alert_on']['unknown_plate']:
                            self.alert_manager.send_alert(
                                title="Unknown Vehicle",
                                message=f"Plate: {plate_number}\nFirst time seen",
                                priority=3,
                                image_path=image_path
                            )
                else:
                    self.logger.warning("No plate detected or confidence too low")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in ANPR loop: {e}")
    
    def save_frame(self, frame, prefix):
        """Save frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        path = Path(self.config['storage']['image_path']) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return str(path)
    
    def start(self):
        """Start all processing threads"""
        self.running = True
        
        detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        anpr_thread = threading.Thread(target=self.anpr_loop, daemon=True)
        anpr_thread.start()
        self.threads.append(anpr_thread)
        
        self.logger.info("All threads started. System running...")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
            self.stop()
    
    def stop(self):
        """Stop all threads and cleanup"""
        self.logger.info("Stopping system...")
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=5)
        
        if self.detection_cam:
            self.detection_cam.stop()
        if self.anpr_cam:
            self.anpr_cam.stop()
        
        self.logger.info("System stopped")


if __name__ == "__main__":
    try:
        camera = SecurityCamera()
        camera.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise
