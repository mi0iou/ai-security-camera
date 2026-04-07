#!/usr/bin/env python3
"""
Alert Manager for Security Camera System
Handles ntfy notifications with image support
"""

import requests
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading

class AlertManager:
    def __init__(self, config, database_manager):
        self.config = config
        self.db = database_manager
        self.logger = logging.getLogger('Alerts')
        
        self.ntfy_server = config['ntfy_server']
        self.ntfy_topic = config['ntfy_topic']
        self.send_images = config['send_images']
        self.cooldown = config['cooldown_seconds']
        
        # Track recent alerts to prevent spam
        self.recent_alerts = {}
        self.lock = threading.Lock()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test ntfy server connection"""
        try:
            response = requests.get(f"{self.ntfy_server}/v1/health", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Connected to ntfy server at {self.ntfy_server}")
            else:
                self.logger.warning(f"ntfy server returned status {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to ntfy server: {e}")
            self.logger.info("Alerts will still be attempted but may fail")
    
    def _should_send_alert(self, plate_number):
        """Check if enough time has passed since last alert for this plate"""
        with self.lock:
            now = datetime.now()
            
            if plate_number in self.recent_alerts:
                last_alert = self.recent_alerts[plate_number]
                time_diff = (now - last_alert).total_seconds()
                
                if time_diff < self.cooldown:
                    self.logger.debug(f"Alert cooldown active for {plate_number} "
                                    f"({int(self.cooldown - time_diff)}s remaining)")
                    return False
            
            self.recent_alerts[plate_number] = now
            return True
    
    def send_alert(self, title, message, priority=3, image_path=None, 
                   plate_number=None, tags=None):
        """
        Send alert via ntfy
        
        Args:
            title: Alert title
            message: Alert message
            priority: 1-5 (1=min, 3=default, 5=max)
            image_path: Path to image to attach
            plate_number: Plate number (for cooldown tracking)
            tags: List of emoji tags (e.g., ['warning', 'car'])
        """
        
        # Check cooldown if plate number provided
        if plate_number and not self._should_send_alert(plate_number):
            return False
        
        try:
            url = f"{self.ntfy_server}/{self.ntfy_topic}"
            headers = {
                "Title": title,
                "Priority": str(priority),
                "Tags": ",".join(tags) if tags else "camera,security"
            }
            
            # If image should be included
            if self.send_images and image_path and Path(image_path).exists():
                # ntfy: send image as request body, message goes in header
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                
                headers["Message"] = message.replace('\n', ' | ')
                headers["Filename"] = Path(image_path).name
                
                response = requests.put(
                    url,
                    data=img_data,
                    headers=headers,
                    timeout=30
                )
            else:
                # Simple text notification
                response = requests.post(
                    url,
                    data=message,
                    headers=headers,
                    timeout=10
                )
            
            if response.status_code == 200:
                self.logger.info(f"Alert sent: {title}")
                return True
            else:
                self.logger.error(f"Alert failed with status {response.status_code}: "
                                f"{response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False
    
    def send_image_alert(self, title, message, image_path, priority=3):
        """
        Send alert with image using ntfy's attachment feature
        This uploads the image separately for better quality
        """
        try:
            url = f"{self.ntfy_server}/{self.ntfy_topic}"
            
            # First, send the image
            with open(image_path, 'rb') as img_file:
                img_response = requests.put(
                    url,
                    data=img_file,
                    headers={
                        "Filename": Path(image_path).name,
                        "Content-Type": "image/jpeg"
                    },
                    timeout=30
                )
            
            if img_response.status_code != 200:
                self.logger.warning(f"Image upload failed: {img_response.status_code}")
            
            # Then send the message with reference to the image
            response = requests.post(
                url,
                data=message,
                headers={
                    "Title": title,
                    "Priority": str(priority),
                    "Tags": "camera,security",
                    "Attach": img_response.text if img_response.status_code == 200 else ""
                },
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error sending image alert: {e}")
            return False
    
    def send_test_alert(self):
        """Send a test notification"""
        return self.send_alert(
            title="Security Camera Test",
            message=f"System is operational at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            priority=1,
            tags=["white_check_mark"]
        )
    
    def send_blacklist_alert(self, plate_number, owner_name, image_path=None):
        """Send high-priority blacklist alert"""
        return self.send_alert(
            title="⚠️ BLACKLISTED VEHICLE DETECTED",
            message=f"Plate: {plate_number}\nOwner: {owner_name}\n"
                   f"Time: {datetime.now().strftime('%H:%M:%S')}",
            priority=5,
            image_path=image_path,
            plate_number=plate_number,
            tags=["warning", "rotating_light", "car"]
        )
    
    def send_unknown_alert(self, plate_number, image_path=None):
        """Send alert for unknown vehicle"""
        return self.send_alert(
            title="Unknown Vehicle",
            message=f"Plate: {plate_number}\n"
                   f"First time seen\n"
                   f"Time: {datetime.now().strftime('%H:%M:%S')}",
            priority=3,
            image_path=image_path,
            plate_number=plate_number,
            tags=["information_source", "car"]
        )
    
    def send_person_alert(self, count=1, image_path=None):
        """Send alert for person detection"""
        return self.send_alert(
            title="Person Detected",
            message=f"Count: {count}\n"
                   f"Time: {datetime.now().strftime('%H:%M:%S')}",
            priority=4,
            image_path=image_path,
            tags=["walking", "person"]
        )
    
    def send_system_alert(self, message, priority=3):
        """Send system status alert"""
        return self.send_alert(
            title="System Alert",
            message=message,
            priority=priority,
            tags=["computer", "gear"]
        )
    
    def cleanup_old_alerts(self):
        """Remove old alerts from tracking (older than cooldown period)"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.cooldown * 2)
            
            # Remove old entries
            to_remove = [
                plate for plate, timestamp in self.recent_alerts.items()
                if timestamp < cutoff
            ]
            
            for plate in to_remove:
                del self.recent_alerts[plate]
            
            if to_remove:
                self.logger.debug(f"Cleaned up {len(to_remove)} old alert entries")
