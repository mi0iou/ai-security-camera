#!/usr/bin/env python3
import sys
import time
import numpy as np
from hailo_detector import HailoDetector

hef = sys.argv[1] if len(sys.argv) > 1 else 'models/yolov8m.hef'
print(f"Benchmarking {hef}...")

detector = HailoDetector(hef, confidence_threshold=0.5)
test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    _ = detector.detect(test_frame)

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    detections = detector.detect(test_frame)
end = time.time()

fps = iterations / (end - start)
latency = (end - start) / iterations * 1000

print(f"\n{'='*50}")
print(f"  {hef}  (Hailo-8)")
print(f"{'='*50}")
print(f"  FPS:      {fps:.2f}")
print(f"  Latency:  {latency:.2f} ms")
print(f"{'='*50}\n")
