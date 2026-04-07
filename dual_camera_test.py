#!/usr/bin/env python3
"""
Dual Camera Test Viewer (Browser Version)
Shows IMX296 (detection) and IMX477 (ANPR) feeds side by side in your browser
Access at http://<pi-ip>:5001

Press Ctrl+C to stop.
"""

import cv2
import time
import threading
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

app = Flask(__name__)

# Global frame storage
frame0_jpeg = None
frame1_jpeg = None
lock = threading.Lock()


def camera_loop():
    """Capture from both cameras and encode as JPEG"""
    global frame0_jpeg, frame1_jpeg

    print("Initializing cameras...")

    cam0 = Picamera2(0)
    cfg0 = cam0.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    cam0.configure(cfg0)
    cam0.start()
    time.sleep(1)
    print("  Camera 0 (IMX296) ready")

    cam1 = Picamera2(1)
    cfg1 = cam1.create_preview_configuration(
        main={"size": (4056, 3040), "format": "RGB888"}
    )
    cam1.configure(cfg1)
    cam1.start()
    time.sleep(1)
    print("  Camera 1 (IMX477) ready")

    print("Both cameras streaming. Open http://<pi-ip>:5001")

    while True:
        try:
            raw0 = cam0.capture_array()
            raw1 = cam1.capture_array()

            # Resize for preview
            disp0 = cv2.resize(raw0, (960, 540))
            disp1 = cv2.resize(raw1, (960, 540))

            # Label each feed
            cv2.putText(disp0, "IMX296 (Detection)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp0, f"{raw0.shape[1]}x{raw0.shape[0]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(disp1, "IMX477 (ANPR)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp1, f"{raw1.shape[1]}x{raw1.shape[0]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Crosshairs for alignment check
            for d in [disp0, disp1]:
                cx, cy = 480, 270
                cv2.line(d, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 1)
                cv2.line(d, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 1)

            # Encode to JPEG
            _, jpg0 = cv2.imencode('.jpg', disp0, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _, jpg1 = cv2.imencode('.jpg', disp1, [cv2.IMWRITE_JPEG_QUALITY, 75])

            with lock:
                frame0_jpeg = jpg0.tobytes()
                frame1_jpeg = jpg1.tobytes()

            time.sleep(0.05)  # ~20 FPS

        except Exception as e:
            print(f"Camera error: {e}")
            time.sleep(1)


def generate_stream(cam_id):
    """MJPEG generator for a given camera"""
    while True:
        with lock:
            frame = frame0_jpeg if cam_id == 0 else frame1_jpeg

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
        else:
            time.sleep(0.2)


@app.route('/feed0')
def feed0():
    return Response(generate_stream(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/feed1')
def feed1():
    return Response(generate_stream(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Dual Camera Test</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #111; color: #fff; font-family: monospace; padding: 20px; }
        h1 { text-align: center; margin-bottom: 20px; color: #0ff; }
        .feeds { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .cam { flex: 1; min-width: 400px; max-width: 960px; }
        .cam h2 { text-align: center; margin-bottom: 8px; font-size: 1rem; color: #8f8; }
        .cam img { width: 100%; border: 1px solid #333; border-radius: 8px; }
        .note { text-align: center; margin-top: 20px; color: #888; font-size: 0.9rem; }
    </style>
</head>
<body>
    <h1>Dual Camera Test Viewer</h1>
    <div class="feeds">
        <div class="cam">
            <h2>Camera 0 &mdash; IMX296 (Detection)</h2>
            <img src="/feed0" alt="IMX296 feed">
        </div>
        <div class="cam">
            <h2>Camera 1 &mdash; IMX477 (ANPR)</h2>
            <img src="/feed1" alt="IMX477 feed">
        </div>
    </div>
    <p class="note">Check: field of view overlap, focus, colour accuracy. Stop with Ctrl+C on the Pi.</p>
</body>
</html>
    ''')


if __name__ == '__main__':
    # Start camera thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("=" * 50)
    print("  Dual Camera Test Viewer")
    print("  Open http://<pi-ip>:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
