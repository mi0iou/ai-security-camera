# Raspberry Pi 5 + Hailo-8L Setup Guide

Complete setup guide for running the AI Security Camera on Raspberry Pi 5 with Hailo-8L acceleration.

## Hardware Required

- Raspberry Pi 5 (8GB recommended)
- Hailo-8L M.2 AI Accelerator
- M.2 HAT for Pi 5 (e.g., Pimoroni NVMe Base or official Pi M.2 HAT+)
- External SSD (recommended for better performance and longer storage retention)
- IMX296 Global Shutter Camera + 6mm lens (detection, ~55° FOV)
- IMX477 HQ Camera + 16mm lens (ANPR, ~22° FOV) — optional
- Good quality power supply (27W USB-C PD)

## Step 1: Flash the OS

1. Download and install [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Choose **Raspberry Pi OS (64-bit)** — Debian Trixie based
3. Select your external SSD as the target
4. Click the gear icon to pre-configure:
   - Set hostname (e.g., `aicamera`)
   - Enable SSH
   - Set username and password
   - Configure WiFi (if needed)
5. Flash and boot the Pi

## Step 2: Initial Configuration

```bash
sudo raspi-config
```

Set the following:
- **Advanced Options → PCIe Speed → Gen 3** (required for Hailo performance)
- **Interface Options → VNC → Enable** (optional, for remote desktop)

Reboot:
```bash
sudo reboot
```

## Step 3: Update System

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

## Step 4: Install Hailo Support

The easiest way is using the `hailo-all` metapackage:

```bash
sudo apt install hailo-all -y
sudo reboot
```

This installs:
- HailoRT runtime
- PCIe driver
- TAPPAS framework
- Python bindings
- rpicam-apps integration

Verify installation:
```bash
hailortcli fw-control identify
```

You should see your Hailo-8L device info including firmware version and serial number.

## Step 5: Install Python Dependencies

```bash
# Core packages
sudo apt install -y python3-pip python3-opencv python3-picamera2

# Python packages (note: --break-system-packages required on Trixie)
pip install -r requirements.txt --break-system-packages
```

This installs Flask, PyYAML, Ultralytics (YOLO), EasyOCR (for ANPR), and other dependencies. EasyOCR will download its English text recognition model (~100MB) on first run.

## Step 6: Clone the Project

```bash
cd ~
git clone https://github.com/yourusername/ai-security-camera.git
cd ai-security-camera
```

## Step 7: Obtain the Hailo Model

You'll need a YOLOv8s model compiled for Hailo-8L (`yolov8s.hef`). Options:

1. **Download pre-compiled** from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) (requires Hailo developer account)
2. **Compile your own** using the Hailo Dataflow Compiler (see Compiling Custom HEF Models below)

Place the model in the `models/` directory:
```bash
mkdir -p models
# Copy your yolov8s.hef file here
```

## Step 8: Configure

```bash
cp config_example.yaml config.yaml
nano config.yaml
```

Key settings to check:
- **Camera indices** — verify with `rpicam-hello --list-cameras`
- **ntfy server/topic** — for push notifications
- **classes_to_detect** — `null` for all 80 COCO classes, or a list like `[0, 2, 5, 7]` (person, car, bus, truck)
- **anpr.plate_region** — set to `"uk"`, `"us"`, or `"eu"` to match your local plate format
- **detection_log_cooldown** — seconds between database entries per class (prevents spam when objects sit in frame; default 30s)

## Step 9: Test the Cameras

> **Note:** On Debian Trixie, use `rpicam-*` commands, not `libcamera-*`

```bash
# List available cameras
rpicam-hello --list-cameras

# Test detection camera (5 second preview)
rpicam-hello --camera 0 -t 5000

# Test ANPR camera (if using dual cameras)
rpicam-hello --camera 1 -t 5000
```

### Dual Camera Test Viewer

For checking camera alignment, focus, and colour accuracy between the two cameras:

```bash
python3 dual_camera_test.py
```

Open `http://<pi-ip>:5001` in a browser. Both feeds are shown side by side with crosshairs for alignment checking. The detection camera's wider FOV (~55°) should fully contain the ANPR camera's narrower FOV (~22°). Mount both cameras close together, pointing in the same direction.

### IMX296 Colour Note

The IMX296 Global Shutter camera outputs BGR format despite requesting RGB888 on Debian Trixie. The `frame_buffer.py` handles this correctly by skipping colour conversion. If colours look wrong in the dashboard, ensure your system is fully updated.

## Step 10: Test Detection

```bash
cd ~/ai-security-camera
python3 main.py
```

In another terminal:
```bash
python3 dashboard.py
```

Open a browser to `http://<pi-ip>:5000` to see the dashboard with live video feed and detection overlays.

If ANPR is enabled, you should see plate reads in the log when vehicles are detected. EasyOCR takes approximately 10 seconds per read on the Pi 5 CPU — this runs on a separate thread and does not block detection.

## Step 11: Setup Services

Create the detection service:
```bash
sudo nano /etc/systemd/system/security_camera.service
```

```ini
[Unit]
Description=AI Security Camera Detection
After=network.target

[Service]
Type=simple
User=tom
WorkingDirectory=/home/tom/ai_security_camera
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Create the dashboard service:
```bash
sudo nano /etc/systemd/system/camera_dashboard.service
```

```ini
[Unit]
Description=Security Camera Dashboard
After=network.target security_camera.service

[Service]
Type=simple
User=tom
WorkingDirectory=/home/tom/ai_security_camera
ExecStart=/usr/bin/python3 dashboard.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

> **Note:** Change `User=tom` and `WorkingDirectory` to match your username and project path.

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable security_camera camera_dashboard
sudo systemctl start security_camera camera_dashboard
```

## Step 12: Setup ntfy (Optional)

For push notifications, you can either use the public ntfy.sh service or run your own server.

### Option A: Use public ntfy.sh
Set your config to use `https://ntfy.sh` and choose a unique topic name.

### Option B: Run local ntfy server
```bash
sudo apt install ntfy -y
sudo systemctl enable ntfy
sudo systemctl start ntfy
```

Then use `http://localhost` as your ntfy server in config.yaml.

Install the ntfy app on your phone (available on [F-Droid](https://f-droid.org/packages/io.heckel.ntfy/) and Google Play) and subscribe to your topic for push notifications.

## Step 13: Manage Known Plates (Optional)

Use the CLI tool to add known and blacklisted plates:

```bash
# Add a known plate
python3 manage_plates.py add "ABC 1234" "John Smith" --vehicle "Blue Ford Focus" --type known

# Add a blacklisted plate
python3 manage_plates.py add "XYZ 9999" "Banned Vehicle" --type blacklist

# List all plates
python3 manage_plates.py list

# Import from CSV
python3 manage_plates.py import plates.csv
```

Known plates are shown with their owner name on the dashboard. Blacklisted plates trigger high-priority alerts.

## Compiling Custom HEF Models

If you need to compile your own models:

1. You'll need a separate Linux machine (x86_64) with the Hailo Dataflow Compiler
2. Install the Hailo Software Suite from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
3. Convert your model: ONNX → HAR → HEF

Basic steps:
```bash
# On your x86_64 machine with Hailo DFC installed
hailo parser onnx yolov8s.onnx
hailo optimize yolov8s.har
hailo compiler yolov8s_optimized.har
```

See Hailo documentation for detailed instructions.

## Troubleshooting

### Hailo device not found
```bash
# Check PCIe
lspci | grep Hailo

# Check driver loaded
lsmod | grep hailo

# Reinstall driver
sudo apt install --reinstall hailort-pcie-driver
sudo reboot
```

### Camera not detected
```bash
# Check cameras (use rpicam on Trixie, not libcamera)
rpicam-hello --list-cameras

# Check for conflicts
sudo fuser /dev/video*
```

### Poor performance
- Ensure PCIe is set to Gen 3 in raspi-config
- Check thermal throttling: `vcgencmd measure_temp`
- Use an SSD instead of SD card

### Dashboard shows "Connecting..."
Main detection service must be running:
```bash
sudo systemctl status security_camera
journalctl -u security_camera -f
```

### Colours look wrong on IMX296
The Global Shutter camera outputs BGR on Debian Trixie despite requesting RGB888. The frame_buffer.py handles this by skipping the colour swap. Ensure your system is fully updated.

### ANPR not reading plates
1. Check camera alignment using `python3 dual_camera_test.py` — both cameras must point the same direction
2. Vehicles at the edges of the detection frame may fall outside the ANPR camera's narrower FOV
3. Check `journalctl -u security_camera -f` for ANPR log messages
4. The `pin_memory` PyTorch warning in the logs is harmless and can be ignored

### Database growing too large
Increase `detection_log_cooldown` in config.yaml (default 30 seconds) to reduce how often the same object type is logged. Adjust `retention_days` to control how long events are kept. The database auto-cleans old records.

## Development Environment

This project works well with VS Code and the Remote-SSH extension for editing files directly on the Pi.

## System Info Reference

This project was developed and tested on:

```
OS: Debian GNU/Linux 13 (trixie)
Kernel: 6.12+
Python: 3.13
HailoRT: 4.20+
```

## Useful Commands

```bash
# Service management
sudo systemctl start security_camera camera_dashboard
sudo systemctl stop camera_dashboard security_camera
sudo systemctl restart security_camera
sudo systemctl status security_camera

# View logs
journalctl -u security_camera -f
journalctl -u camera_dashboard -f

# Check Hailo
hailortcli fw-control identify
hailortcli monitor

# Check cameras
rpicam-hello --list-cameras

# Check temperatures
vcgencmd measure_temp

# Check disk usage
df -h

# Plate management
python3 manage_plates.py list
python3 manage_plates.py stats
python3 manage_plates.py events --hours 24
```
