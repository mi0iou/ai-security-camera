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
                # EasyOCR runs PyTorch on the CPU. By default PyTorch grabs every
                # core, which makes it fight the detection loop for CPU and inflated
                # live reads to ~9s (vs ~1s of actual work). Pinning to a single
                # thread removes that contention; the OCR input is tiny (a capped
                # plate crop) so it does not need parallelism. Benchmarked: threads>1
                # gave no speed-up and occasionally spiked under load.
                try:
                    import torch
                    torch.set_num_threads(1)
                    self.logger.info("Pinned PyTorch to 1 thread (avoids detection-loop contention)")
                except Exception as e:
                    self.logger.warning(f"Could not set torch thread count: {e}")
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
        
        # Max width for OCR input — controls speed/accuracy tradeoff.
        # Used by the geometric/centre fallback paths, where we don't know the
        # plate's pixel height. 800px keeps fallback-crop text readable.
        self.max_ocr_width = 800

        # Max PLATE height (px) for the tight plate_bbox path. The plate detector
        # gives us the plate's exact pixel height, so we cap the crop so the plate
        # is at most this tall — never upscaling, so distant (already-small) plates
        # are untouched while oversized close plates are shrunk. This is the main
        # live speed lever: close plates were arriving at EasyOCR ~800px wide (~9s
        # under load); capped to a ~96px plate they read correctly in ~1s.
        # Benchmarked across near/far plates: 96px held 100% read accuracy and was
        # the fastest cap tested; larger caps were slower with no accuracy gain.
        self.max_plate_height = config.get('max_plate_height', 96)
    
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
    
    def _resize_to_plate_height(self, image, plate_px_h):
        """Downscale a tight plate crop so the plate is at most max_plate_height
        tall. One-directional: a plate already shorter (distant vehicle) is passed
        through untouched, preserving its limited detail; only oversized close
        plates are shrunk. plate_px_h is the plate's height in the ORIGINAL frame
        (py2 - py1), i.e. the plate within this crop, not the padded crop height.
        """
        if plate_px_h <= 0 or plate_px_h <= self.max_plate_height:
            return image  # already at/under the ceiling — leave it alone
        scale = self.max_plate_height / plate_px_h  # < 1.0, downscale only
        h, w = image.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.logger.debug(f"Plate-height cap: {w}x{h} -> {new_w}x{new_h} "
                          f"(plate {plate_px_h}px -> ~{self.max_plate_height}px)")
        return resized

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
    
    def read_plate(self, image, vehicle_bbox=None, plate_bbox=None):
        """
        Main method to read a number plate from an image.
        
        Args:
            image: Full camera frame (RGB)
            vehicle_bbox: Optional [x1, y1, x2, y2] in this camera's coordinates.
                         Dramatically speeds up processing by cropping first.
            plate_bbox: Optional [x1, y1, x2, y2] tight plate region in this
                         camera's coordinates, e.g. from the Hailo plate detector.
                         When supplied, EasyOCR is fed a tight crop of just the
                         plate (with small padding), skipping the geometric
                         lower-50% guess entirely. Falls back to the geometric
                         crop if the plate_bbox is empty/invalid.
        
        Returns:
            dict with 'plate', 'confidence', 'raw_text' or None
        """
        img_h, img_w = image.shape[:2]
        t_start = time.time()
        
        # Step 1: Choose the search region.
        # Priority: explicit plate_bbox (tight) > vehicle_bbox (geometric) > centre crop.
        search_image = None
        crop_path = "none"  # which crop fed OCR — surfaced at INFO for diagnosis
        plate_px_h = None   # plate height in original frame; set on the tight path

        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            pw = px2 - px1
            ph = py2 - py1

            # Small padding around the plate so OCR has context / margin
            pad_x = int(pw * 0.15)
            pad_y = int(ph * 0.25)
            crop_x1 = max(0, px1 - pad_x)
            crop_y1 = max(0, py1 - pad_y)
            crop_x2 = min(img_w, px2 + pad_x)
            crop_y2 = min(img_h, py2 + pad_y)

            candidate = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if candidate.size == 0:
                self.logger.warning("Plate crop was empty, falling back to geometric crop")
            else:
                search_image = candidate
                crop_path = "plate_bbox (tight)"
                plate_px_h = ph  # plate height in original frame, for height cap
                self.logger.debug(f"Plate crop: {search_image.shape[1]}x{search_image.shape[0]} "
                                 f"from {img_w}x{img_h}")

        # Geometric fallback: crop to vehicle region if bbox provided
        if search_image is None and vehicle_bbox is not None:
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
                crop_path = "full_frame"
            else:
                crop_path = "vehicle_bbox (geometric)"
                self.logger.debug(f"Vehicle crop: {search_image.shape[1]}x{search_image.shape[0]} "
                                 f"from {img_w}x{img_h}")

        # Final fallback: no usable plate_bbox or vehicle_bbox — centre-lower area
        if search_image is None:
            # likely plate position with telephoto
            cx, cy = img_w // 2, int(img_h * 0.6)
            crop_w, crop_h = img_w // 3, img_h // 6
            search_image = image[
                max(0, cy - crop_h):min(img_h, cy + crop_h),
                max(0, cx - crop_w):min(img_w, cx + crop_w)
            ]
            self.logger.debug(f"Centre crop: {search_image.shape[1]}x{search_image.shape[0]}")
            crop_path = "centre_crop"
        
        # Step 2: Resize for fast OCR.
        # Tight plate path: cap by PLATE HEIGHT (we know it from the detector), which
        # shrinks oversized close plates while leaving distant small plates intact.
        # Fallback paths: no known plate height, so keep the longest-edge width cap.
        if plate_px_h is not None:
            ocr_image = self._resize_to_plate_height(search_image, plate_px_h)
        else:
            ocr_image = self._resize_for_ocr(search_image)
        
        # Step 3: Run OCR
        if self.method == 'easyocr':
            ocr_results = self._run_easyocr(ocr_image)
        else:
            ocr_results = self._run_tesseract(ocr_image)
        
        elapsed = time.time() - t_start
        self.logger.debug(f"OCR took {elapsed:.1f}s, found {len(ocr_results)} text regions")
        # INFO-level diagnostic: which crop fed OCR, its size, and how long OCR took.
        # plate_bbox (tight) should be ~1s; vehicle_bbox (geometric) is the slow ~8s path.
        self.logger.info(f"OCR path={crop_path} input={ocr_image.shape[1]}x{ocr_image.shape[0]} "
                         f"time={elapsed:.1f}s regions={len(ocr_results)}")
        
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
