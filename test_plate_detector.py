#!/usr/bin/env python3
"""
Integration test for license_plate_detector.hef
Plumbing check only: confirms the HEF loads, runs, and returns plate boxes.
Saves an annotated copy with boxes drawn so the bbox placement can be eyeballed.
NOT part of the live pipeline. Stop services before running (releases Hailo device).
"""
import sys
import os
import logging
import cv2

from hailo_detector import HailoDetector

logging.basicConfig(level=logging.INFO)

HEF = '/home/tom/ai_security_camera/models/license_plate_detector.hef'

def main(image_path):
    frame_bgr = cv2.imread(image_path)          # BGR, as loaded
    if frame_bgr is None:
        print(f"Could not read {image_path}")
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # wrapper expects RGB

    detector = HailoDetector(
        hef_path=HEF,
        confidence_threshold=0.25,
        class_names=['license_plate'],
    )

    detections = detector.detect(frame_rgb)
    print(f"\n{len(detections)} detection(s):")
    for d in detections:
        print(f"  {d['class_name']}  conf={d['confidence']:.3f}  bbox={d['bbox']}")

    # Draw boxes on the original BGR image (so colours are correct when saved)
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_annotated{ext}"
    cv2.imwrite(out_path, frame_bgr)
    print(f"\nAnnotated image saved: {out_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 test_plate_detector.py <image_with_a_plate.jpg>")
        sys.exit(1)
    main(sys.argv[1])
