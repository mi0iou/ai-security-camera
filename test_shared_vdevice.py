#!/usr/bin/env python3
"""
Step 2b bench: prove yolov8s + license-plate detector coexist on ONE Hailo device
via a shared scheduler-enabled VDevice, across two threads.

This is the empirical test of the open question: does the round-robin scheduler
hold yolov8s throughput while the plate detector fires from another thread, using
the (older) InferVStreams blocking API?

Standalone — does NOT touch main.py. Stop services first to free the device:
    sudo systemctl stop security_camera camera_dashboard

Usage:
    python3 test_shared_vdevice.py <test_image.jpg> [seconds]

Reports yolov8s FPS in three phases:
    baseline (plate thread idle) -> contended (plate thread firing) -> recovery
so you can see the scheduler's impact directly.
"""
import sys
import time
import threading
import logging
import cv2

from hailo_detector import HailoDetector

logging.basicConfig(level=logging.WARNING)   # quiet; we print our own numbers
log = logging.getLogger("shared_vdev_test")
log.setLevel(logging.INFO)

YOLO_HEF  = '/home/tom/ai_security_camera/models/yolov8s.hef'
PLATE_HEF = '/home/tom/ai_security_camera/models/license_plate_detector.hef'


def main(image_path, run_seconds=15):
    frame = cv2.imread(image_path)
    if frame is None:
        log.error(f"Could not read {image_path}")
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- one shared, scheduler-enabled VDevice for BOTH models ---
    log.info("Creating shared scheduler-enabled VDevice...")
    vdevice = HailoDetector.create_shared_vdevice()

    log.info("Configuring yolov8s onto shared device...")
    yolo = HailoDetector(hef_path=YOLO_HEF, confidence_threshold=0.6, vdevice=vdevice)

    log.info("Configuring plate detector onto shared device...")
    plate = HailoDetector(hef_path=PLATE_HEF, confidence_threshold=0.25,
                          class_names=['license_plate'], vdevice=vdevice)

    log.info("Both models configured on one VDevice. Starting threads.\n")

    stop = threading.Event()
    plate_active = threading.Event()      # gate the plate thread on/off for phases
    plate_calls = [0]

    def plate_worker():
        while not stop.is_set():
            if plate_active.is_set():
                plate.detect(frame_rgb)
                plate_calls[0] += 1
                time.sleep(0.2)           # ~5 reads/sec, mimics intermittent ANPR
            else:
                time.sleep(0.02)

    pt = threading.Thread(target=plate_worker, daemon=True)
    pt.start()

    def measure_yolo_fps(duration, label):
        n = 0
        t0 = time.time()
        while time.time() - t0 < duration:
            yolo.detect(frame_rgb)
            n += 1
        dt = time.time() - t0
        fps = n / dt if dt > 0 else 0
        print(f"  {label:<28} {fps:6.1f} FPS   ({n} frames in {dt:.1f}s)")
        return fps

    third = max(3, run_seconds // 3)
    print("=== yolov8s throughput (shared VDevice, round-robin scheduler) ===")
    plate_active.clear()
    base = measure_yolo_fps(third, "baseline (plate idle):")

    plate_active.set()
    cont = measure_yolo_fps(third, "contended (plate firing):")

    plate_active.clear()
    time.sleep(0.5)
    recov = measure_yolo_fps(third, "recovery (plate idle):")

    stop.set()
    pt.join(timeout=2)

    print(f"\n  plate detector calls during contended phase: {plate_calls[0]}")
    if base > 0:
        drop = 100.0 * (base - cont) / base
        print(f"  yolov8s FPS drop under contention: {drop:.0f}%")
    print("\nInterpretation: a modest drop is expected and fine (device is shared).")
    print("A near-total collapse or errors above = the scheduler/threading path")
    print("needs rework before wiring into main.py.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_shared_vdevice.py <test_image.jpg> [seconds]")
        sys.exit(1)
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(sys.argv[1], secs)
