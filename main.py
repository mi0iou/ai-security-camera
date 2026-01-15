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
                    
                    # Track person detections for this frame
                    person_count = 0
                    person_image_path = None
                    
                    for det in detections:
                        cls = det['class']
                        conf = det['confidence']
                        bbox = det['bbox']
                        class_name = det['class_name']
                        
                        self.logger.info(f"{class_name} detected (class {cls}) with confidence {conf:.2f}")
                        
                        # Save image
                        image_path = None
                        if self.config['storage']['save_detection_frames']:
                            image_path = self.save_frame(frame, f"{class_name}_{cls}")
                        
                        # Log ALL detections as events
                        self.db.log_event(
                            timestamp=datetime.now(),
                            event_type=class_name,
                            plate_number=None,
                            confidence=conf,
                            image_path=image_path
                        )
                        
                        # Track person detections
                        if cls == 0:  # person
                            person_count += 1
                            if person_image_path is None:
                                person_image_path = image_path
                        
                        # Special handling for vehicles - trigger ANPR
                        if cls in [2, 5, 7]:  # car, bus, truck
                            if self.config['anpr']['enabled']:
                                try:
                                    self.anpr_queue.put_nowait({
                                        'timestamp': datetime.now(),
                                        'detection_frame': frame.copy(),
                                        'bbox': bbox,
                                        'class': cls,
                                        'confidence': conf
                                    })
                                except queue.Full:
                                    self.logger.warning("ANPR queue full, skipping frame")
                    
                    # Send person alert if enabled and persons detected
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
                            'detection_count': len(detections)
                        }
                        frame_buffer.publish_frame(frame, detections, stats)
                        last_publish_time = current_time
                
                else:
                    results = self.model(frame, conf=confidence, classes=classes_to_detect, verbose=False)
                    
                    self._last_inference_ms = (time.time() - inference_start) * 1000
                    
                    detections_list = []
                    person_count = 0
                    person_image_path = None
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            # Get class name
                            class_names = result.names
                            class_name = class_names[cls]
                            
                            self.logger.info(f"{class_name} detected (class {cls}) with confidence {conf:.2f}")
                            
                            # Build detection dict for frame buffer
                            detections_list.append({
                                'class': cls,
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            })
                            
                            # Save image
                            image_path = None
                            if self.config['storage']['save_detection_frames']:
                                image_path = self.save_frame(frame, f"{class_name}_{cls}")
                            
                            # Log ALL detections as events
                            self.db.log_event(
                                timestamp=datetime.now(),
                                event_type=class_name,
                                plate_number=None,
                                confidence=conf,
                                image_path=image_path
                            )
                            
                            # Track person detections
                            if cls == 0:  # person
                                person_count += 1
                                if person_image_path is None:
                                    person_image_path = image_path
                            
                            # Special handling for vehicles - trigger ANPR
                            if cls in [2, 5, 7]:
                                if self.config['anpr']['enabled']:
                                    try:
                                        self.anpr_queue.put_nowait({
                                            'timestamp': datetime.now(),
                                            'detection_frame': frame.copy(),
                                            'bbox': xyxy,
                                            'class': cls,
                                            'confidence': conf
                                        })
                                    except queue.Full:
                                        self.logger.warning("ANPR queue full, skipping frame")
                    
                    # Send person alert if enabled and persons detected
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
    
    def anpr_loop(self):
        """ANPR processing loop on IMX477"""
        self.logger.info("ANPR loop started")
        
        while self.running:
            try:
                detection_data = self.anpr_queue.get(timeout=1)
                
                anpr_frame = self.anpr_cam.capture_array()
                
                plate_result = self.anpr.read_plate(anpr_frame)
                
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
                                title="⚠️ BLACKLISTED VEHICLE",
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
