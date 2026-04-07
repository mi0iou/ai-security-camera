#!/usr/bin/env python3
"""
ANPR (Automatic Number Plate Recognition) Module
Uses EasyOCR for reliable text detection in natural scenes.

Strategy:
  1. Crop to vehicle region using mapped bounding box (from detection camera)
  2. Take the lower portion only (plates are on bumpers, not roofs)
  3. Cap the crop at ~800px wide for fast OCR (~2-5s on Pi 5 CPU)
  4. EasyOCR finds and reads text automatically
  5. Validate against regional plate patterns
"""

import cv2
import numpy as np
import re
import logging
import time


class ANPRProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ANPR')
        self.method = config.get('method', 'easyocr')
        
        # Initialize EasyOCR (preferred) or Tesseract (fallback)
        if self.method == 'easyocr':
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False)
                self.logger.info("EasyOCR initialized")
            except ImportError:
                self.logger.error("EasyOCR not installed. Install with: pip install easyocr --break-system-packages")
                raise
        elif self.method == 'tesseract':
            try:
                import pytesseract
                self.reader = pytesseract
                self.logger.info("Tesseract initialized")
            except ImportError:
                self.logger.error("pytesseract not installed")
                raise
        else:
            raise ValueError(f"Unknown ANPR method: {self.method}")
        
        # Plate patterns by region
        self.plate_patterns = {
            'uk': r'^([A-Z]{2}[0-9]{2}\s?[A-Z]{3}|[A-Z]{2,3}\s?[0-9]{2,4})$',
            'us': r'^[A-Z0-9]{2,7}$',
            'eu': r'^[A-Z]{1,3}[-\s]?[0-9]{1,4}[-\s]?[A-Z]{1,3}$',
        }
        
        # Max width for OCR input — controls speed/accuracy tradeoff
        # 800px is fast (~2-5s on Pi 5) while keeping plate text readable
        self.max_ocr_width = 800
    
    def clean_plate_text(self, text):
        """Clean and format plate text"""
        text = re.sub(r'[^A-Z0-9\s-]', '', text.upper())
        text = ' '.join(text.split())
        return text.strip()
    
    def validate_plate(self, text):
        """Validate plate against regional pattern"""
        if not text or len(text) < 4:
            return False
        
        region = self.config.get('plate_region', 'uk')
        pattern = self.plate_patterns.get(region)
        
        if not pattern:
            return bool(re.match(r'^[A-Z0-9\s-]{4,10}$', text))
        
        return bool(re.match(pattern, text))
    
    def _resize_for_ocr(self, image):
        """Resize image so the longest edge is max_ocr_width, if needed"""
        h, w = image.shape[:2]
        if w <= self.max_ocr_width:
            return image
        
        scale = self.max_ocr_width / w
        new_w = self.max_ocr_width
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.logger.debug(f"Resized {w}x{h} -> {new_w}x{new_h} for OCR")
        return resized
    
    def _run_easyocr(self, image):
        """
        Run EasyOCR on image.
        Returns list of (text, confidence) tuples for all detected text.
        """
        try:
            results = self.reader.readtext(image)
            return [(r[1], r[2]) for r in results if r[2] > 0.1]
        except Exception as e:
            self.logger.error(f"EasyOCR error: {e}")
            return []
    
    def _run_tesseract(self, image):
        """Fallback: run Tesseract on image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            config = '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            data = self.reader.image_to_data(gray, config=config,
                                             output_type=self.reader.Output.DICT)
            
            results = []
            words = []
            for i, word in enumerate(data['text']):
                word = word.strip()
                if not word:
                    continue
                conf = int(data['conf'][i]) if str(data['conf'][i]) != '-1' else 0
                if conf > 0:
                    words.append((word, conf / 100.0))
                    if len(word) >= 4:
                        results.append((word, conf / 100.0))
            
            # Also try combining consecutive word pairs
            for i in range(len(words) - 1):
                for sep in ['', ' ']:
                    combined = words[i][0] + sep + words[i+1][0]
                    avg_conf = (words[i][1] + words[i+1][1]) / 2
                    results.append((combined, avg_conf))
            
            return results
        except Exception as e:
            self.logger.error(f"Tesseract error: {e}")
            return []
    
    def read_plate(self, image, vehicle_bbox=None):
        """
        Main method to read a number plate from an image.
        
        Args:
            image: Full camera frame (RGB)
            vehicle_bbox: Optional [x1, y1, x2, y2] in this camera's coordinates.
                         Dramatically speeds up processing by cropping first.
        
        Returns:
            dict with 'plate', 'confidence', 'raw_text' or None
        """
        img_h, img_w = image.shape[:2]
        t_start = time.time()
        
        # Step 1: Crop to vehicle region if bbox provided
        if vehicle_bbox is not None:
            vx1, vy1, vx2, vy2 = vehicle_bbox
            vh = vy2 - vy1
            vw = vx2 - vx1
            
            # Take lower 50% of vehicle (where plates are) with padding
            crop_y1 = max(0, vy1 + int(vh * 0.4))
            crop_y2 = min(img_h, vy2 + int(vh * 0.15))
            crop_x1 = max(0, vx1 - int(vw * 0.1))
            crop_x2 = min(img_w, vx2 + int(vw * 0.1))
            
            search_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if search_image.size == 0:
                self.logger.warning("Vehicle crop was empty, using full frame")
                search_image = image
            else:
                self.logger.debug(f"Vehicle crop: {search_image.shape[1]}x{search_image.shape[0]} "
                                 f"from {img_w}x{img_h}")
        else:
            # No bbox — use centre-lower area (likely plate position with telephoto)
            cx, cy = img_w // 2, int(img_h * 0.6)
            crop_w, crop_h = img_w // 3, img_h // 6
            search_image = image[
                max(0, cy - crop_h):min(img_h, cy + crop_h),
                max(0, cx - crop_w):min(img_w, cx + crop_w)
            ]
            self.logger.debug(f"Centre crop: {search_image.shape[1]}x{search_image.shape[0]}")
        
        # Step 2: Resize for fast OCR
        ocr_image = self._resize_for_ocr(search_image)
        
        # Step 3: Run OCR
        if self.method == 'easyocr':
            ocr_results = self._run_easyocr(ocr_image)
        else:
            ocr_results = self._run_tesseract(ocr_image)
        
        elapsed = time.time() - t_start
        self.logger.debug(f"OCR took {elapsed:.1f}s, found {len(ocr_results)} text regions")
        
        # Step 4: Find best plate match
        best_result = None
        best_conf = 0
        
        for raw_text, conf in ocr_results:
            cleaned = self.clean_plate_text(raw_text)
            
            if self.validate_plate(cleaned) and conf > best_conf:
                self.logger.info(f"Plate read: '{cleaned}' conf={conf:.2f} ({elapsed:.1f}s)")
                best_result = {
                    'plate': cleaned,
                    'confidence': conf,
                    'raw_text': raw_text
                }
                best_conf = conf
        
        # Also try combining adjacent results (handles split reads)
        if not best_result and len(ocr_results) >= 2:
            for i in range(len(ocr_results) - 1):
                for sep in ['', ' ']:
                    combined = ocr_results[i][0] + sep + ocr_results[i+1][0]
                    avg_conf = (ocr_results[i][1] + ocr_results[i+1][1]) / 2
                    cleaned = self.clean_plate_text(combined)
                    
                    if self.validate_plate(cleaned) and avg_conf > best_conf:
                        self.logger.info(f"Combined plate: '{cleaned}' conf={avg_conf:.2f} ({elapsed:.1f}s)")
                        best_result = {
                            'plate': cleaned,
                            'confidence': avg_conf,
                            'raw_text': combined
                        }
                        best_conf = avg_conf
        
        if not best_result:
            self.logger.debug(f"No plate detected ({elapsed:.1f}s)")
        
        return best_result
