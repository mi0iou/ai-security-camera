#!/usr/bin/env python3
"""
Step 1 bench test: full ANPR crop-wiring chain on a static image.

  plate detector (Hailo)  ->  highest-confidence plate bbox
                          ->  ANPRProcessor.read_plate(plate_bbox=...)
                          ->  EasyOCR (unchanged)

Standalone: instantiates ONLY the plate detector (no yolov8s), so there is no
VDevice contention here. Shared-VDevice work is deferred to Step 2.

Proves the functional path only. Accuracy is not the point (level-0 HEF).

Usage:
    python3 test_anpr_pipeline.py <image_with_a_plate.jpg>

Stop services first to release the Hailo device:
    sudo systemctl stop security_camera camera_dashboard
"""
import sys
import logging
import cv2
import yaml

from hailo_detector import HailoDetector
from anpr_module import ANPRProcessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("anpr_pipeline_test")

HEF = '/home/tom/ai_security_camera/models/license_plate_detector.hef'
CONFIG = '/home/tom/ai_security_camera/config.yaml'


def main(image_path):
    # --- load image (BGR from disk -> RGB for the pipeline) ---
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        log.error(f"Could not read {image_path}")
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # --- ANPR processor (real config, so EasyOCR + plate_region match production) ---
    with open(CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    anpr = ANPRProcessor(config['anpr'])

    # --- plate detector (own VDevice; fine because yolov8s is not running here) ---
    plate_detector = HailoDetector(
        hef_path=HEF,
        confidence_threshold=0.25,
        class_names=['license_plate'],
    )

    # --- Step A: detect plate(s) on the full frame ---
    detections = plate_detector.detect(frame_rgb)
    log.info(f"Plate detector returned {len(detections)} detection(s)")
    for d in detections:
        log.info(f"  {d['class_name']} conf={d['confidence']:.3f} bbox={d['bbox']}")

    # --- Step B: pick highest-confidence plate, pass its bbox to read_plate ---
    plate_bbox = None
    if detections:
        best = max(detections, key=lambda d: d['confidence'])
        plate_bbox = best['bbox']
        log.info(f"Using plate_bbox={plate_bbox} (conf={best['confidence']:.3f})")
    else:
        log.info("No plate detected -> read_plate will fall back to geometric/centre crop")

    # --- Step C: OCR via the existing, unchanged read_plate call ---
    result = anpr.read_plate(frame_rgb, plate_bbox=plate_bbox)

    print("\n=== RESULT ===")
    if result:
        print(f"  plate     : {result['plate']}")
        print(f"  confidence: {result['confidence']:.3f}")
        print(f"  raw_text  : {result['raw_text']}")
    else:
        print("  No plate read.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 test_anpr_pipeline.py <image_with_a_plate.jpg>")
        sys.exit(1)
    main(sys.argv[1])
