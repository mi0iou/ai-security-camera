# AI Security Camera System

**Raspberry Pi 5 + Hailo-8L powered security camera with real-time object detection, ANPR, and smart alerts**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red.svg)
![Hailo](https://img.shields.io/badge/accelerator-Hailo--8L-orange.svg)

A complete AI-powered security camera system that runs entirely on edge hardware. Uses YOLOv8 accelerated by the Hailo-8L AI accelerator for real-time object detection at 20+ FPS, with optional ANPR (Automatic Number Plate Recognition) for vehicle monitoring.

## Features

- **Real-time Object Detection** - YOLOv8s running on Hailo-8L at 20+ FPS
- **Dual Camera Support** - Separate cameras for detection and ANPR
- **Automatic Number Plate Recognition** - Detect and log vehicle plates
- **Smart Alerts** - Push notifications via ntfy for people, unknown/blacklisted vehicles
- **Web Dashboard** - Live video feed with detection overlays and statistics
- **Event Logging** - SQLite database for all detections and plate sightings
- **Auto-start on Boot** - Systemd services for headless operation
- **Low Latency** - All processing on-device, no cloud required

## Hardware Requirements

| Component | Model | Purpose |
|-----------|-------|---------|
| Single Board Computer | Raspberry Pi 5 (8GB recommended) | Main compute |
| AI Accelerator | Hailo-8L | Neural network inference |
| Detection Camera | IMX296 (Global Shutter) | Object detection |
| ANPR Camera (optional) | IMX477 (HQ Camera) | License plate capture |
| Storage | External SSD recommended | OS and recordings |

## Quick Start

For complete setup instructions including OS installation and Hailo configuration, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**.

### Prerequisites

- Raspberry Pi OS (Debian Trixie, 64-bit)
- Hailo-8L with `hailo-all` package installed
- Python 3.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/mi0iou/ai-security-camera.git
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

You'll need to obtain or compile a `yolov8s.hef` model for the Hailo-8L. Place it in the `models/` directory. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for details.

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

## Project Structure

```
ai-security-camera/
├── main.py                 # Main detection loop
├── dashboard.py            # Flask web dashboard
├── hailo_detector.py       # Hailo-8L inference wrapper
├── frame_buffer.py         # Shared frame buffer for dashboard
├── anpr_module.py          # License plate recognition
├── database_manager.py     # SQLite database operations
├── alert_manager.py        # ntfy push notifications
├── config.yaml             # Configuration file (create from example)
├── config_example.yaml     # Example configuration
├── models/
│   └── yolov8s.hef         # Hailo model (you provide)
├── database/
│   └── security.db         # SQLite database (auto-created)
├── images/                 # Saved detection frames
└── logs/                   # Log files
```

## Configuration

Key settings in `config.yaml`:

```yaml
detection:
  use_hailo: true
  hailo_model_path: "models/yolov8s.hef"
  confidence_threshold: 0.5
  classes_to_detect: [0, 2, 5, 7]  # person, car, bus, truck or use 'null' to detect ALL classes

alerts:
  ntfy_server: "http://localhost"  # or https://ntfy.sh
  ntfy_topic: "your-topic"
  cooldown_seconds: 60
  alert_on:
    person_detected: true
    unknown_plate: true
    blacklisted_plate: true
```

## Performance

Tested on Raspberry Pi 5 (8GB) with Hailo-8L:

| Metric | Value |
|--------|-------|
| Detection FPS | 20-25 FPS |
| Inference Time | ~40ms |
| End-to-end Latency | <100ms |
| CPU Usage | ~30% |
| Power Consumption | ~8W total |

## Technical Notes

- **Hailo coordinate format:** The Hailo NMS postprocessor outputs `[y1, x1, y2, x2]`, not the standard `[x1, y1, x2, y2]`. This is handled in `hailo_detector.py`.
- **Letterboxing:** Input is letterboxed to 640x640 maintaining aspect ratio to prevent detection distortion.
- **Frame sharing:** Uses `/dev/shm` for fast cross-process frame sharing between detection and dashboard.
- **IMX296 colour format:** The Global Shutter camera outputs BGR directly despite requesting RGB888 format. The frame buffer handles this without conversion.

## Troubleshooting

### Bounding boxes are misaligned
Ensure you're using the latest `hailo_detector.py` which handles the `[y1, x1, y2, x2]` coordinate format.

### Dashboard shows "Connecting..."
The detection service must be running:
```bash
sudo systemctl status security_camera
```

### Hailo device not found
```bash
hailortcli fw-control identify
```
If this fails, check PCIe is set to Gen 3 in `raspi-config`.

### Colours look wrong (red/blue swapped)
This is a known issue with the IMX296 Global Shutter camera. The `frame_buffer.py` has been updated to handle the BGR output format correctly. If colours still appear swapped, ensure you have the latest version.

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more troubleshooting tips.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The included YOLOv8 model is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics.

## Acknowledgments

- [Hailo](https://hailo.ai/) for the edge AI accelerator
- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for the Pi 5
- [ntfy](https://ntfy.sh/) for simple push notifications
