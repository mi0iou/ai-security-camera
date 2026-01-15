#!/usr/bin/env python3
"""
ANPR (Automatic Number Plate Recognition) Module
Supports EasyOCR and Tesseract
"""

import cv2
import numpy as np
import re
import logging

class ANPRProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ANPR')
        self.method = config['method']
        
        # Initialize OCR engine
        if self.method == 'easyocr':
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False)
                self.logger.info("EasyOCR initialized")
            except ImportError:
                self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise
        elif self.method == 'tesseract':
            try:
                import pytesseract
                self.reader = pytesseract
                self.logger.info("Tesseract initialized")
            except ImportError:
                self.logger.error("pytesseract not installed. Install with: pip install pytesseract")
                raise
        else:
            raise ValueError(f"Unknown ANPR method: {self.method}")
        
        # Plate patterns by region
        self.plate_patterns = {
            'uk': r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$',  # UK format: AB12 CDE
            'us': r'^[A-Z0-9]{2,7}$',  # US format: varies by state
            'eu': r'^[A-Z]{1,3}[-\s]?[0-9]{1,4}[-\s]?[A-Z]{1,3}$',  # EU format
        }
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def detect_plate_region(self, image):
        """
        Detect potential plate regions using contours
        Returns list of cropped plate regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edges = cv2.Canny(blur, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_regions = []
        
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # Check if contour has 4 points (potential rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (typical plate ratio is 2:1 to 5:1)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                    # Extract region
                    plate_region = image[y:y+h, x:x+w]
                    plate_regions.append({
                        'region': plate_region,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio
                    })
        
        return plate_regions
    
    def read_plate_easyocr(self, image):
        """Read plate using EasyOCR"""
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Run OCR
            results = self.reader.readtext(processed)
            
            if not results:
                return None
            
            # Get best result
            best_result = max(results, key=lambda x: x[2])  # Sort by confidence
            text = best_result[1]
            confidence = best_result[2]
            
            # Clean text
            text = self.clean_plate_text(text)
            
            # Validate against pattern
            if self.validate_plate(text):
                return {
                    'plate': text,
                    'confidence': confidence,
                    'raw_text': best_result[1]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"EasyOCR error: {e}")
            return None
    
    def read_plate_tesseract(self, image):
        """Read plate using Tesseract"""
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Configure Tesseract for plate recognition
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Run OCR
            text = self.reader.image_to_string(processed, config=custom_config)
            
            # Get confidence (if available)
            data = self.reader.image_to_data(processed, config=custom_config, output_type=self.reader.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0
            
            # Clean text
            text = self.clean_plate_text(text)
            
            # Validate against pattern
            if self.validate_plate(text):
                return {
                    'plate': text,
                    'confidence': avg_confidence,
                    'raw_text': text
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Tesseract error: {e}")
            return None
    
    def clean_plate_text(self, text):
        """Clean and format plate text"""
        # Remove special characters except spaces and hyphens
        text = re.sub(r'[^A-Z0-9\s-]', '', text.upper())
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Common OCR mistakes
        replacements = {
            'O': '0',  # O to 0 in number positions
            'I': '1',  # I to 1
            'Z': '2',  # Z to 2
            'S': '5',  # S to 5
            'B': '8',  # B to 8
        }
        
        # Apply replacements intelligently based on position
        # This is simplified - you may want region-specific logic
        
        return text.strip()
    
    def validate_plate(self, text):
        """Validate plate against regional pattern"""
        if not text or len(text) < 2:
            return False
        
        region = self.config.get('plate_region', 'uk')
        pattern = self.plate_patterns.get(region)
        
        if not pattern:
            # If no pattern, accept any alphanumeric 2-10 chars
            return bool(re.match(r'^[A-Z0-9\s-]{2,10}$', text))
        
        return bool(re.match(pattern, text))
    
    def read_plate(self, image):
        """
        Main method to read plate from image
        Tries multiple approaches for best results
        """
        results = []
        
        # Try on full image first
        if self.method == 'easyocr':
            result = self.read_plate_easyocr(image)
        else:
            result = self.read_plate_tesseract(image)
        
        if result and result['confidence'] >= self.config['min_confidence']:
            return result
        
        # If failed, try detecting plate regions first
        plate_regions = self.detect_plate_region(image)
        
        for region_data in plate_regions[:3]:  # Try top 3 regions
            if self.method == 'easyocr':
                result = self.read_plate_easyocr(region_data['region'])
            else:
                result = self.read_plate_tesseract(region_data['region'])
            
            if result and result['confidence'] >= self.config['min_confidence']:
                return result
        
        return None
