#!/usr/bin/env python3
"""
Diagnostic: determine the true channel order of IMX477 frames on this Pi.

Captures ONE frame from the ANPR camera and saves it two ways:
  *_as_is.jpg     - array written straight to imwrite (correct IF array is BGR)
  *_swapped.jpg   - array run through a R<->B swap before imwrite
                    (correct IF array is RGB)

Open both. The red Isuzu should look RED in exactly one of them:
  - red in *_as_is.jpg   -> IMX477 delivers BGR  (same quirk as IMX296)
  - red in *_swapped.jpg -> IMX477 delivers RGB  (no quirk; current save_frame is right)

Stop main.py first so the camera is free.

Usage: python3 test_imx477_channels.py
"""
import time
import cv2
import yaml
from picamera2 import Picamera2

CONFIG = '/home/tom/ai_security_camera/config.yaml'

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

idx = cfg['cameras']['anpr']['index']
res = tuple(cfg['cameras']['anpr']['resolution'])

cam = Picamera2(idx)
cam.configure(cam.create_preview_configuration(main={"size": res, "format": "RGB888"}))
cam.start()
time.sleep(2)                      # let AE/AWB settle
frame = cam.capture_array()
cam.stop()
cam.close()

print(f"Captured IMX477 frame: shape={frame.shape}, dtype={frame.dtype}")

cv2.imwrite('imx477_as_is.jpg', frame)
cv2.imwrite('imx477_swapped.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("Saved: imx477_as_is.jpg  and  imx477_swapped.jpg")
print("Open both. Whichever shows the vehicle in its TRUE colour reveals the order:")
print("  red in *_as_is   -> array is BGR")
print("  red in *_swapped -> array is RGB")
