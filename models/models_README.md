# Models

Pre-compiled YOLOv8 model for Hailo-8L AI accelerator.

## Included Files

| File | Size | Description |
|------|------|-------------|
| `yolov8s.hef` | ~25MB | YOLOv8s compiled for Hailo-8L |

## Model Details

- **Architecture:** YOLOv8s (small)
- **Input Size:** 640x640
- **Classes:** 80 (COCO dataset)
- **Target Hardware:** Hailo-8L (13 TOPS)
- **Compiled With:** Hailo Dataflow Compiler

## Usage

The model is automatically loaded by `hailo_detector.py`. Ensure the path in `config.yaml` matches:

```yaml
detection:
  hailo_model_path: "models/yolov8s.hef"
```

## COCO Classes Used for Security

| ID | Class | Used For |
|----|-------|----------|
| 0 | person | Person detection alerts |
| 2 | car | Vehicle detection / ANPR trigger |
| 5 | bus | Vehicle detection / ANPR trigger |
| 7 | truck | Vehicle detection / ANPR trigger |

## Compiling Your Own Models

If you need to compile custom models, you'll need:

1. A Linux x86_64 machine (not the Pi)
2. Hailo Software Suite from [Hailo Developer Zone](https://hailo.ai/developer-zone/)
3. The Hailo Model Zoo

Basic process:
```
.pt (PyTorch) → .onnx (ONNX) → .har (Hailo Archive) → .hef (Hailo Executable)
```

This is resource-intensive and complex. The included pre-compiled .hef is recommended for most users.

## License

YOLOv8 models are released under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics.
