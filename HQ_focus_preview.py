#!/usr/bin/env python3
"""
IMX477 Focus Preview
====================
Standalone live MJPEG preview of the ANPR camera (IMX477) for checking
manual-lens focus from a browser on your laptop.

This opens the camera directly via Picamera2, so the main system must NOT
be running at the same time (it holds the IMX477 open):

    sudo systemctl stop security_camera
    sudo systemctl stop camera_dashboard   # optional, frees port 5000 too
    python3 focus_preview.py

Then open from your laptop browser:

    http://aicamera:8000/

Ctrl+C to quit. Restart the services when done:

    sudo systemctl start security_camera
    sudo systemctl start camera_dashboard

Notes:
- Reads the ANPR camera index from config.yaml (falls back to 1).
- The IMX477's native 4056x3040 is downscaled for a smooth stream; focus
  sharpness is preserved well enough at preview resolution to judge by eye.
  Use the browser zoom (Ctrl/Cmd +) on the live image for a closer look.
- Plain preview only, no overlays.
"""

import io
import sys
import time
import socket
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import yaml
from picamera2 import Picamera2

# ---- Config ---------------------------------------------------------------

HTTP_PORT = 8000
# Preview stream resolution (downscaled from the sensor's native res).
# 1280x960 keeps the 4:3 IMX477 aspect ratio and streams smoothly.
PREVIEW_SIZE = (1280, 960)
JPEG_QUALITY = 80
TARGET_FPS = 15

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("focus_preview")


def load_anpr_index(config_path="config.yaml"):
    """Read the ANPR (IMX477) camera index from config.yaml; default to 1."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return int(cfg["cameras"]["anpr"]["index"])
    except Exception as e:
        log.warning("Could not read ANPR index from %s (%s); using 1", config_path, e)
        return 1


# ---- Camera ---------------------------------------------------------------

class FocusCamera:
    def __init__(self, index):
        self.cam = Picamera2(index)
        # Request a downscaled preview directly so we don't pull full 4056x3040
        # frames every cycle. RGB888 to match the rest of the project; note the
        # IMX477 delivers BGR channel order in practice on Trixie, which is fine
        # here because cv2.imencode expects BGR anyway.
        config = self.cam.create_preview_configuration(
            main={"size": PREVIEW_SIZE, "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(2)  # let AE/AWB settle
        log.info("IMX477 started on index %d at %dx%d", index, *PREVIEW_SIZE)

    def jpeg(self):
        frame = self.cam.capture_array()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()

    def stop(self):
        try:
            self.cam.stop()
        except Exception:
            pass


# ---- HTTP server ----------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IMX477 Focus Preview</title>
<style>
  body{margin:0;background:#0a0a0f;color:#8888aa;font-family:sans-serif;
       display:flex;flex-direction:column;align-items:center}
  h1{font-size:1rem;font-weight:600;margin:12px 0;color:#00f0ff}
  img{max-width:100%;height:auto;border:1px solid #2a2a3a}
  p{font-size:0.8rem;margin:8px}
</style></head>
<body>
  <h1>IMX477 Focus Preview</h1>
  <img src="/stream" alt="live feed">
  <p>Adjust the lens until the image is sharp. Browser zoom (Ctrl/Cmd +) for a closer look.</p>
</body></html>"""


def make_handler(camera):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # quiet the default per-request logging

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                frame_interval = 1.0 / TARGET_FPS
                try:
                    while True:
                        t0 = time.time()
                        jpeg = camera.jpeg()
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        dt = time.time() - t0
                        if dt < frame_interval:
                            time.sleep(frame_interval - dt)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed
                return

            self.send_error(404)

    return Handler


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<pi-ip>"


def main():
    index = load_anpr_index()
    try:
        camera = FocusCamera(index)
    except Exception as e:
        log.error("Failed to open IMX477 on index %d: %s", index, e)
        log.error("Is main.py / the security_camera service still running? "
                  "Stop it first: sudo systemctl stop security_camera")
        sys.exit(1)

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), make_handler(camera))
    ip = local_ip()
    log.info("Focus preview running. Open in your browser:")
    log.info("    http://%s:%d/   (or http://aicamera:%d/)", ip, HTTP_PORT, HTTP_PORT)
    log.info("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        server.shutdown()
        camera.stop()


if __name__ == "__main__":
    main()
