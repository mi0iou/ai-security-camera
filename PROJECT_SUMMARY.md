# AI Security Camera - Project Summary

## Overview

A Raspberry Pi 5 + Hailo-8 based security camera with:

- **YOLOv8 object detection** at 40+ FPS using Hailo-8 acceleration
- **Dual camera support** — IMX296 (6mm, ~55° FOV) for detection, IMX477 (16mm, ~22° FOV) for ANPR
- **ANPR pipeline** — detection camera spots vehicles → bbox mapped to ANPR camera via angular FOV ratio → EasyOCR reads plate → validated against UK/NI/US/EU patterns
- **Web dashboard** with live MJPEG video feed, detection overlays, detected plates sidebar, and event history
- **Cross-process frame sharing** via `/dev/shm` so dashboard doesn't impact detection
- **ntfy push notifications** for alerts to phone
- **SQLite logging** of all events with per-class cooldown
- **Systemd services** for auto-start on boot

## Key Files

| File | Purpose |
|------|---------|
| main.py | Detection loop, dual-camera ANPR triggering with bbox mapping, database logging |
| dashboard.py | Flask web UI with live video, detected plates sidebar, events table (port 5000) |
| hailo_detector.py | Hailo inference wrapper (handles [y1,x1,y2,x2] coordinate order) |
| frame_buffer.py | Cross-process frame sharing via /dev/shm (handles IMX296 BGR output) |
| database_manager.py | SQLite operations including detected plates query |
| alert_manager.py | ntfy push notifications (text and image attachments) |
| anpr_module.py | EasyOCR-based license plate recognition with regional validation |
| manage_plates.py | CLI tool for known/blacklisted plate management |
| dual_camera_test.py | Side-by-side dual camera test viewer (port 5001) |
| live_viewer.py | Local OpenCV detection viewer for dev/debug |
| benchmark_hailo.py | Hailo-8 inference benchmarking |
| config.yaml | All configuration |
| config_example.yaml | Example config with comments (committed to repo) |

## Services

- **Detection service:** `security_camera.service`
- **Dashboard service:** `camera_dashboard.service`

## Technical Notes

### Hailo Coordinate Format
The Hailo NMS postprocessor outputs coordinates as `[y1, x1, y2, x2]` not standard `[x1, y1, x2, y2]`. This is handled in `hailo_detector.py` with proper coordinate transformation.

### Letterboxing
Input frames are letterboxed to 640x640 maintaining aspect ratio to prevent detection distortion. Coordinates are properly scaled back to original image dimensions.

### Dual Camera Bbox Mapping
`_map_bbox_to_anpr()` in main.py converts detection camera bounding boxes (IMX296, 6mm, ~55° FOV, 1920×1080) to ANPR camera coordinates (IMX477, 16mm, ~22° FOV, 4056×3040) using pixels-per-degree angular mapping with 30% padding for alignment tolerance.

### ANPR Pipeline
Vehicle detected → bbox mapped to ANPR camera → IMX477 captures 4056×3040 → NPU plate detector (YOLOv8n, single class, shared Hailo-8) localises the plate → crop downscaled so the plate is ≤ `max_plate_height` px tall (default 96) → EasyOCR reads text → validate against plate regex → log to SQLite + ntfy alert. If the plate detector misses, falls back to geometric crop (vehicle lower half, 800px wide). Runs on a separate thread.

Live read time ~1.7s on the tight path (was ~9s before optimisation). Two levers: `_resize_to_plate_height` caps the plate at 96px (one-directional — shrinks close plates, never upscales distant ones), and PyTorch is pinned to one thread at ANPR init so EasyOCR stops contending with the detection loop. The 96px default was chosen by `benchmark_ocr_contention.py` (100% accuracy across near/far plates, fastest cap). `read_plate` takes optional `plate_bbox`; logs `OCR path=... input=WxH time=Xs regions=N` at INFO. `main.py` writes a `<frame>.json` sidecar (plate bbox + confidences) next to each saved ANPR frame so the contention benchmark can reproduce the exact live crop without re-localising.

### Frame Buffer
Uses `/dev/shm` for fast cross-process communication. Dashboard reads frames at ~10 FPS while detection runs at full speed independently.

### IMX296 Colour Format
The Global Shutter camera outputs BGR format despite requesting RGB888. The frame_buffer.py handles this by passing frames through without colour conversion.

### Debian Trixie
On Trixie, use `rpicam-*` commands instead of `libcamera-*`.

---

## Development Prompt

Copy and paste this to start a new chat:

---

I'm continuing development on my AI Security Camera project. Here's the context:

**Hardware:** Raspberry Pi 5 (8GB) + Hailo-8 AI accelerator (26 TOPS, M.2) + IMX296 Global Shutter camera (detection, 6mm lens) + IMX477 HQ camera (ANPR, 16mm lens)

**Project location:** /home/tom/ai_security_camera

**Key components:**
- main.py — Main detection loop using YOLOv8 on Hailo-8, maps vehicle bboxes to ANPR camera via angular FOV ratio, runs the NPU plate detector, triggers EasyOCR plate reads, logs to SQLite, sends ntfy alerts
- dashboard.py — Flask web dashboard with live MJPEG feed, detected plates sidebar, events table (port 5000)
- hailo_detector.py — Hailo inference wrapper with letterboxing and coordinate transformation (Hailo outputs [y1,x1,y2,x2] not [x1,y1,x2,y2])
- frame_buffer.py — Cross-process frame sharing using /dev/shm for dashboard preview (handles IMX296 BGR output)
- database_manager.py — SQLite operations including detected plates query
- alert_manager.py — ntfy push notifications (text; image attachments via PUT)
- anpr_module.py — EasyOCR-based plate recognition with UK/NI regex validation
- manage_plates.py — CLI plate management tool
- dual_camera_test.py — Side-by-side camera test viewer (port 5001)

**Current status:** Working end-to-end system with:
- 40+ FPS detection with proper bounding boxes
- Two NPU detectors (YOLOv8s object + YOLOv8n single-class plate) sharing one Hailo-8 via a round-robin scheduler VDevice
- Live web dashboard with video feed, detected plates sidebar, per-class counts
- ANPR: detection camera spots vehicles → bbox mapped to ANPR camera → NPU plate detector localises tight crop → crop capped to ~96px plate height → EasyOCR reads (~1.7s live) → validated → logged + alerted
- Dual camera test viewer at :5001 for alignment/focus checking
- Auto-start via systemd services (security_camera, camera_dashboard)
- Push notifications via local ntfy server (text only)
- Per-class detection logging with 30s cooldown
- ANPR trigger cooldown (30s per vehicle class)

**Technical notes:**
- Two NPU detectors share one Hailo-8 via HailoDetector.create_shared_vdevice() (ROUND_ROBIN); shared mode skips own VDevice creation and manual network_group.activate()
- Plate detector loaded outside the try/except in setup_yolo — missing/broken plate HEF hard-fails startup (plate stage always on)
- ANPR OCR latency: tight crop capped to max_plate_height px (default 96, one-directional downscale) + torch.set_num_threads(1) at ANPR init → ~9s to ~1.7s live, no accuracy loss. Geometric fallback keeps 800px width cap.
- main.py writes a <frame>.json sidecar (plate bbox + confidences) next to saved ANPR frames for benchmark_ocr_contention.py
- Hailo NMS postprocessor outputs coordinates as [y1,x1,y2,x2] not standard [x1,y1,x2,y2]
- Letterboxing required to maintain aspect ratio (1920×1080 → 640×640)
- Frame buffer uses /dev/shm for fast cross-process communication
- IMX296 and IMX477 both output BGR despite RGB888 request on Trixie — no conversion; save_frame writes as-is
- On Debian Trixie, use rpicam-* commands not libcamera-*
- Dual camera bbox mapping uses pixels-per-degree angular ratio with 30% padding
- EasyOCR is CPU-only, pinned to 1 thread, runs on a separate thread; read_plate takes optional plate_bbox

I code in Python. I want to: [DESCRIBE WHAT YOU WANT TO DO NEXT]
