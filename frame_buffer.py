#!/usr/bin/env python3
"""
Shared Frame Buffer - Cross-Process Version
Uses file-based sharing so main.py and dashboard.py can communicate
"""

import threading
import time
import cv2
import numpy as np
from pathlib import Path
import json
import os
import tempfile


class FrameBuffer:
    """
    Cross-process frame buffer using file system for sharing.
    Detection system writes frames to a temp file, dashboard reads them.
    """
    
    def __init__(self):
        # Use /dev/shm for faster RAM-based file sharing (Linux tmpfs)
        # Falls back to /tmp if /dev/shm doesn't exist
        if os.path.exists('/dev/shm'):
            self._base_path = Path('/dev/shm/security_cam')
        else:
            self._base_path = Path(tempfile.gettempdir()) / 'security_cam'
        
        self._base_path.mkdir(parents=True, exist_ok=True)
        
        self._frame_path = self._base_path / 'frame.jpg'
        self._meta_path = self._base_path / 'meta.json'
        self._lock = threading.Lock()
        
    def publish_frame(self, frame_rgb, detections=None, stats=None):
        """
        Publish a frame from the detection system.
        Called by main.py detection loop.
        
        Args:
            frame_rgb: RGB frame from camera
            detections: List of detection dicts with bbox, class, confidence
            stats: Dict with fps, inference_ms, etc.
        """
        try:
            frame_bgr = frame_rgb  # IMX296 already delivers BGR
            
            # Draw bounding boxes
            if detections:
                colors = {
                    0: (255, 0, 0),      # person - blue
                    2: (0, 255, 0),      # car - green
                    5: (0, 165, 255),    # bus - orange
                    7: (0, 0, 255),      # truck - red
                }
                default_color = (255, 255, 0)
                
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    color = colors.get(det['class'], default_color)
                    
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                    cv2.putText(frame_bgr, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add overlay info
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if stats:
                cv2.putText(frame_bgr, f"FPS: {stats.get('fps', 0):.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Detections: {stats.get('detection_count', 0)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Inference: {stats.get('inference_ms', 0):.0f}ms", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add "LIVE" indicator
            cv2.putText(frame_bgr, "LIVE", (frame_bgr.shape[1] - 80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Resize for preview (smaller = less disk I/O)
            preview_width = 960
            preview_height = 540
            frame_preview = cv2.resize(frame_bgr, (preview_width, preview_height))
            
            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame_preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            # Write to temp file (atomic write using rename)
            temp_frame_path = self._base_path / 'frame_tmp.jpg'
            temp_meta_path = self._base_path / 'meta_tmp.json'
            
            with open(temp_frame_path, 'wb') as f:
                f.write(jpeg.tobytes())
            
            # Build live per-class counts from current detections
            live_counts = {}
            if detections:
                for det in detections:
                    name = det['class_name']
                    live_counts[name] = live_counts.get(name, 0) + 1
            
            # Write metadata
            meta = {
                'timestamp': time.time(),
                'detection_count': len(detections) if detections else 0,
                'stats': stats or {},
                'live_counts': live_counts
            }
            with open(temp_meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Atomic rename
            os.replace(temp_frame_path, self._frame_path)
            os.replace(temp_meta_path, self._meta_path)
                
        except Exception as e:
            print(f"FrameBuffer publish error: {e}")
    
    def get_frame(self):
        """
        Get the latest frame as JPEG bytes.
        Called by dashboard for streaming.
        
        Returns:
            JPEG bytes or None if no frame available
        """
        try:
            if self._frame_path.exists():
                with open(self._frame_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            pass
        return None
    
    def get_frame_age(self):
        """Get age of current frame in seconds"""
        try:
            if self._meta_path.exists():
                with open(self._meta_path, 'r') as f:
                    meta = json.load(f)
                    return time.time() - meta.get('timestamp', 0)
        except:
            pass
        return float('inf')
    
    def get_stats(self):
        """Get current detection stats"""
        try:
            if self._meta_path.exists():
                with open(self._meta_path, 'r') as f:
                    meta = json.load(f)
                    return meta.get('stats', {})
        except:
            pass
        return {}
    
    def get_meta(self):
        """Get full metadata"""
        try:
            if self._meta_path.exists():
                with open(self._meta_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None


# Global instance - each process gets its own but they share via filesystem
frame_buffer = FrameBuffer()

