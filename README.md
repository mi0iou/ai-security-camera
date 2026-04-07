# AI Security Camera System

**Raspberry Pi 5 + Hailo-8L powered security camera with real-time object detection, ANPR, and smart alerts**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red.svg)
![Hailo](https://img.shields.io/badge/accelerator-Hailo--8L-orange.svg)

A complete AI-powered security camera system that runs entirely on edge hardware. Uses YOLOv8 accelerated by the Hailo-8L AI accelerator for real-time object detection at 20+ FPS, with dual-camera ANPR (Automatic Number Plate Recognition) for vehicle monitoring. All processing happens on-device — no cloud required.

## Features

- **Real-time Object Detection** — YOLOv8s on Hailo-8L at 20+ FPS with ~40ms inference
- **Live Web Dashboard** — MJPEG video feed with detection overlays, live per-class counts, detected plates sidebar, and event history
- **Dual Camera Support** — IMX296 Global Shutter (6mm, ~55° FOV) for detection, IMX477 HQ Camera (16mm, ~22° FOV) for ANPR
- **Automatic Number Plate Recognition** — Detection camera spots vehicles → bounding box mapped to ANPR camera via angular FOV ratio → high-res capture → EasyOCR reads plate → validated against regional patterns
- **Smart Alerts** — Push notifications via ntfy with configurable priorities and cooldowns
- **Per-class Detection Logging** — Configurable cooldown prevents database spam while keeping the live feed responsive
- **Event Logging** — SQLite database with 24-hour statistics, detection breakdowns, and retention policies
- **Cross-process Frame Sharing** — `/dev/shm` based buffer isolates dashboard from detection performance
- **Dual Camera Test Viewer** — Side-by-side MJPEG viewer on port 5001 for checking camera alignment, focus, and colour
- **CLI Plate Management** — Add, remove, search, import/export known and blacklisted plates
- **Auto-start on Boot** — Systemd services for fully headless operation
- **Fallback Mode** — Automatically falls back to CPU-based YOLO if Hailo is unavailable

## Hardware Requirements

| Component | Model | Purpose |
|-----------|-------|---------|
| Single Board Computer | Raspberry Pi 5 (8GB recommended) | Main compute |
| AI Accelerator | Hailo-8L (M.2) | Neural network inference |
| Detection Camera | IMX296 (Global Shutter) + 6mm lens | Object detection (~55° FOV) |
| ANPR Camera (optional) | IMX477 (HQ Camera) + 16mm lens | License plate capture (~22° FOV) |
| M.2 HAT | Pimoroni NVMe Base or official Pi M.2 HAT+ | Hailo-8L mounting |
| Storage | External SSD recommended | OS and recordings |
| Power Supply | 27W USB-C PD | Stable power |

## Quick Start

For complete setup instructions including OS installation and Hailo configuration, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**.

### Prerequisites

- Raspberry Pi OS (Debian Trixie, 64-bit)
- Hailo-8L with `hailo-all` package installed
- Python 3.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-security-camera.git
cd ai-security-camera

# Install Python dependencies
pip install -r requirements.txt --break-system-packages

# Configure
cp config_example.yaml config.yaml
nano config.yaml

# Test
python3 main.py
```

### Pre-compiled Model

You'll need a `yolov8s.hef` model compiled for the Hailo-8L. Place it in the `models/` directory. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for details.

## Usage

### Manual Start

```bash
# Start detection system
python3 main.py

# In another terminal, start dashboard
python3 dashboard.py
```

Access the dashboard at `http://<pi-ip>:5000`

### Service Management

```bash
# Start/stop services
sudo systemctl start security_camera camera_dashboard
sudo systemctl stop camera_dashboard security_camera

# View logs
journalctl -u security_camera -f

# Check status
systemctl status security_camera camera_dashboard
```

### Plate Management

```bash
# Add a known plate
python3 manage_plates.py add "ABC 1234" "John Smith" --type known

# Add a blacklisted plate
python3 manage_plates.py add "XYZ 9999" "Banned" --type blacklist

# List all plates
python3 manage_plates.py list

# View statistics
python3 manage_plates.py stats

# Import/export CSV
python3 manage_plates.py import plates.csv
python3 manage_plates.py export backup.csv
```

### Dual Camera Test

Use the built-in test viewer to check camera alignment, focus, and colour before deploying:

```bash
python3 dual_camera_test.py
```

Open `http://<pi-ip>:5001` to see both camera feeds side by side with crosshairs for alignment checking.

## Architecture

```
┌─────────────────────┐     /dev/shm      ┌──────────────────────┐
│     main.py         │ ──────────────────▶│    dashboard.py      │
│  Detection Loop     │   frame + meta     │  Flask Web UI :5000  │
│  ANPR Processing    │                    │  MJPEG Stream        │
│  Alert Dispatch     │                    │  Live Counts API     │
│  DB Logging         │                    │  Detected Plates API │
└────────┬────────────┘                    └──────────┬───────────┘
         │                                            │
         ▼                                            ▼
  ┌──────────────┐                           ┌──────────────┐
  │  SQLite DB   │◀──────────────────────────│  Read-only   │
  │  security.db │                           │  DB queries  │
  └──────────────┘                           └──────────────┘
```

The detection process (`main.py`) owns the cameras and Hailo device. The dashboard (`dashboard.py`) runs as a separate process, reading frames from shared memory and querying the database read-only. This ensures the web UI never impacts detection performance.

### ANPR Pipeline

When a vehicle is detected by the IMX296 camera, the system maps the bounding box from the detection camera's coordinate space to the ANPR camera's coordinate space using an angular FOV ratio (pixels-per-degree). The IMX477 then captures a full-resolution 4056×3040 frame, crops to the lower half of the mapped vehicle region (where plates are), resizes to 800px wide, and runs EasyOCR. Detected text is validated against regional plate patterns (UK/NI, US, EU) and logged to the database.

## Dashboard

The web dashboard provides:

- **Live MJPEG video feed** with bounding box overlays drawn server-side
- **In Frame Now** — real-time count of objects currently visible, updated every 0.5s
- **Per-class breakdown** — sidebar showing count of each object type in the current frame
- **Detected Plates** — auto-refreshing sidebar showing plates seen in the last 24 hours with confidence, sighting count, and known/blacklist status
- **24-hour statistics** — unique plates seen, blacklist alerts, pulled from the database
- **Recent Events table** — auto-refreshing log of all detection events
- **Reset Counters** — clears event history from the database

### Split Data Architecture

The dashboard uses two data sources for responsiveness:

- **Live counts** (objects in frame, people now, per-class breakdown) come from frame buffer metadata updated every 0.5 seconds — no database queries needed
- **Cumulative stats** (unique plates, alerts, events, detected plates) query the SQLite database at longer intervals (5–10 seconds)

## Project Structure

```
ai-security-camera/
├── main.py                 # Main detection loop with ANPR and alerts
├── dashboard.py            # Flask web dashboard (port 5000)
├── hailo_detector.py       # Hailo-8L inference with letterboxing
├── frame_buffer.py         # Cross-process frame sharing via /dev/shm
├── anpr_module.py          # License plate recognition (EasyOCR)
├── database_manager.py     # SQLite database operations
├── alert_manager.py        # ntfy push notifications
├── manage_plates.py        # CLI tool for plate database management
├── live_viewer.py          # Local OpenCV detection viewer (dev/debug)
├── dual_camera_test.py     # Side-by-side dual camera test viewer (port 5001)
├── benchmark_hailo.py      # Hailo-8L inference benchmarking tool
├── config.yaml             # Configuration (create from example)
├── config_example.yaml     # Example configuration with comments
├── requirements.txt        # Python dependencies
├── models/
│   ├── yolov8s.hef         # Hailo model (you provide)
│   └── README.md           # Model details and compilation notes
├── database/
│   └── security.db         # SQLite database (auto-created)
├── images/                 # Saved detection/ANPR frames
└── logs/                   # Log files
```

## Configuration

Key settings in `config.yaml`:

```yaml
detection:
  use_hailo: true
  hailo_model_path: "models/yolov8s.hef"
  confidence_threshold: 0.5
  classes_to_detect: null            # null = all 80 COCO classes

anpr:
  enabled: true
  method: "easyocr"                  # EasyOCR for plate recognition
  min_confidence: 0.5
  plate_region: "uk"                 # "uk", "us", or "eu"

alerts:
  ntfy_server: "http://localhost"    # or https://ntfy.sh
  ntfy_topic: "your-topic"
  cooldown_seconds: 30
  alert_on:
    person_detected: true
    unknown_plate: true
    blacklisted_plate: true

performance:
  detection_interval: 0.05           # Target ~20 FPS
  detection_log_cooldown: 30         # Seconds between DB logs per class
```

The `detection_log_cooldown` setting controls how often the same object type is logged to the database. This prevents thousands of duplicate entries when an object sits in frame, while still showing everything on the live feed. Set to `0` to log every detection.

## Performance

Tested on Raspberry Pi 5 (8GB) with Hailo-8L:

| Metric | Value |
|--------|-------|
| Detection FPS | 20–25 FPS |
| Inference Time | ~40ms |
| End-to-end Latency | <100ms |
| ANPR Read Time | ~10s per plate (EasyOCR on CPU) |
| CPU Usage | ~30% |
| Power Consumption | ~8W total |

ANPR runs on a separate thread so plate reading does not block detection. The ~10s read time is typical for EasyOCR without GPU acceleration; the ANPR trigger cooldown prevents repeated captures of the same vehicle while it sits in frame.

## Technical Notes

- **Hailo coordinate format:** The Hailo NMS postprocessor outputs `[y1, x1, y2, x2]`, not the standard `[x1, y1, x2, y2]`. This is handled in `hailo_detector.py`.
- **Letterboxing:** Input is letterboxed from 1920×1080 to 640×640 maintaining aspect ratio to prevent detection distortion. Coordinates are properly scaled back to original image space.
- **Dual camera bbox mapping:** Detection bounding boxes are mapped from the IMX296 (6mm, ~55° FOV) to the IMX477 (16mm, ~22° FOV) using pixels-per-degree angular ratios with 30% padding for alignment tolerance.
- **Frame sharing:** Uses `/dev/shm` for fast cross-process frame sharing between detection and dashboard processes.
- **IMX296 colour format:** The Global Shutter camera outputs BGR directly despite requesting RGB888 format on Debian Trixie. The frame buffer handles this without conversion.
- **Debian Trixie:** Use `rpicam-*` commands (e.g. `rpicam-hello`), not the older `libcamera-*` commands.
- **EasyOCR pin_memory warning:** A harmless PyTorch warning about `pin_memory` appears because EasyOCR runs without GPU. It can be safely ignored or suppressed.

## Troubleshooting

### Bounding boxes are misaligned
Ensure you're using the latest `hailo_detector.py` which handles the `[y1, x1, y2, x2]` coordinate format from the Hailo NMS postprocessor.

### Dashboard shows "Connecting..."
The detection service must be running first:
```bash
sudo systemctl status security_camera
journalctl -u security_camera -f
```

### Hailo device not found
```bash
hailortcli fw-control identify
```
If this fails, check PCIe is set to Gen 3 in `raspi-config` → Advanced Options → PCIe Speed.

### Colours look wrong (red/blue swapped)
This is a known issue with the IMX296 Global Shutter camera on Debian Trixie. The `frame_buffer.py` skips colour conversion since the camera outputs BGR directly. Ensure your system is fully updated:
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### ANPR not reading plates
Check camera alignment using `python3 dual_camera_test.py` at `http://<pi-ip>:5001`. The ANPR camera's narrower FOV (~22°) means vehicles detected at the edges of the detection camera's wider FOV (~55°) may fall outside the ANPR frame. Ensure both cameras point in the same direction and that vehicles pass through the centre of the detection frame.

### Database grows too large
Adjust `retention_days` in config.yaml and increase `detection_log_cooldown` to reduce logging frequency. The system auto-cleans old records based on the retention policy.

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more troubleshooting tips.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The included YOLOv8 model is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics.

## Acknowledgments

- [Hailo](https://hailo.ai/) for the edge AI accelerator
- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for the Pi 5
- [ntfy](https://ntfy.sh/) for simple push notifications
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for number plate text recognition
