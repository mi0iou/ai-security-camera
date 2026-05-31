#!/usr/bin/env python3
"""
Option B experiment: does feeding the plate detector + EasyOCR true RGB
(instead of the BGR the live system currently passes) improve accuracy?

Runs ONE saved frame through the full detect -> crop -> OCR chain TWICE:
  BGR path : array as the live system currently sees it (camera delivers BGR,
             passed straight to detector/OCR)
  RGB path : same array with R<->B swapped to true RGB

Saved plate_*.jpg are stored BGR by imwrite, so cv2.imread gives BGR == exactly
what the live camera array contains. Valid stand-in for live capture.

Compares: detector confidence, bbox, and OCR result/confidence for each path.

Stop main.py first (frees the Hailo device).

Usage: python3 test_channel_experiment.py <saved_plate_frame.jpg>
"""
import sys
import logging
import cv2
import yaml

from hailo_detector import HailoDetector
from anpr_module import ANPRProcessor

logging.basicConfig(level=logging.WARNING)   # quiet; we print our own comparison

HEF = '/home/tom/ai_security_camera/models/license_plate_detector.hef'
CONFIG = '/home/tom/ai_security_camera/config.yaml'


def run_chain(label, image, detector, anpr):
    dets = detector.detect(image)
    if not dets:
        print(f"\n[{label}] detector: NO plate found")
        return
    best = max(dets, key=lambda d: d['confidence'])
    print(f"\n[{label}] detector: conf={best['confidence']:.3f} bbox={best['bbox']}")
    result = anpr.read_plate(image, plate_bbox=best['bbox'])
    if result:
        print(f"[{label}] OCR     : '{result['plate']}' conf={result['confidence']:.3f}")
    else:
        print(f"[{label}] OCR     : no read")


def main(image_path):
    bgr = cv2.imread(image_path)          # BGR == live camera array order
    if bgr is None:
        print(f"Could not read {image_path}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # true RGB

    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    anpr = ANPRProcessor(config['anpr'])
    detector = HailoDetector(hef_path=HEF, confidence_threshold=0.25,
                             class_names=['license_plate'])

    print("=== Channel-order experiment ===")
    print("(higher detector conf + higher OCR conf = better)")
    run_chain("BGR (current live)", bgr, detector, anpr)
    run_chain("RGB (swapped)     ", rgb, detector, anpr)
    print("\nNote: OCR time will be slow standalone too; this test is about")
    print("ACCURACY (confidence), not speed.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 test_channel_experiment.py <saved_plate_frame.jpg>")
        sys.exit(1)
    main(sys.argv[1])
