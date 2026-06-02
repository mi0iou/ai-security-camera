# Models

Pre-compiled YOLOv8 models for the Hailo-8 AI accelerator. Two HEFs are required,
both compiled for the **Hailo-8** (26 TOPS) — not the Hailo-8L. HEFs are
architecture-specific; an 8L-compiled HEF will not give correct/performant
results on a Hailo-8.

Both `.hef` files are gitignored and not committed to the repository — you must
supply or compile them yourself and place them in this directory.

## Required Files

| File | Description |
|------|-------------|
| `yolov8s.hef` | YOLOv8s object detector (80 COCO classes), compiled for Hailo-8 |
| `license_plate_detector.hef` | Single-class (`license_plate`) YOLOv8n detector, compiled for Hailo-8 |

## Model Details

### Object detector — `yolov8s.hef`

- **Architecture:** YOLOv8s (small)
- **Input size:** 640×640
- **Classes:** 80 (COCO)
- **Target hardware:** Hailo-8 (26 TOPS)
- **Compiled with:** Hailo Dataflow Compiler

### Plate detector — `license_plate_detector.hef`

- **Architecture:** YOLOv8n (nano), single class (`license_plate`)
- **Input size:** 640×640
- **Classes:** 1
- **Target hardware:** Hailo-8 (26 TOPS)
- **Compiled with:** Hailo Dataflow Compiler (see compilation notes below)

Both detectors run concurrently on the single physical Hailo-8 via a shared,
scheduler-enabled VDevice (round-robin). See the main
[README](../README.md) and [SETUP_GUIDE](../SETUP_GUIDE.md) for the runtime
architecture.

## Usage

Both models are loaded by `hailo_detector.py`. Ensure the paths in `config.yaml`
match:

```yaml
detection:
  hailo_model_path: "models/yolov8s.hef"
  plate_model_path: "models/license_plate_detector.hef"
```

## COCO Classes Used for Security (object detector)

| ID | Class | Used For |
|----|-------|----------|
| 0 | person | Person detection alerts |
| 2 | car | Vehicle detection / ANPR trigger |
| 5 | bus | Vehicle detection / ANPR trigger |
| 7 | truck | Vehicle detection / ANPR trigger |

## Compiling Your Own Models

Compilation must be done on a Linux x86_64 machine (not the Pi), using the Hailo
Software Suite from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
and the Hailo Model Zoo. The general pipeline is:

```
.pt (PyTorch) → .onnx (ONNX) → .har (Hailo Archive) → .hef (Hailo Executable)
```

This is resource-intensive and complex. The notes below document how the
production plate detector was built, including the non-obvious pitfalls that cost
time — useful both for rebuilding the plate detector and as a worked example for
compiling the object detector.

## Plate Detector Compilation Notes

These notes describe how the production `license_plate_detector.hef` was compiled.
They are recorded here because the HEF is not committed to the repo, so this is
the only record of how to reproduce it.

### Source model & dataset

- **Model:** Custom-trained YOLOv8n, single class (`license_plate`), exported to
  ONNX (opset 11, 640×640 input). Standard YOLOv8n topology — the parsed HAR
  shows conv1–conv63.
- **Calibration dataset:** `keremberke/license-plate-object-detection` — the v1
  original-images export (8823 images, CC BY 4.0, no augmentation applied). 1200
  images were sampled from the train split and flattened to clean RGB; 1024 were
  used for calibration. Validation/test splits were left untouched for later
  accuracy evaluation (see [INT8 Quantization Accuracy](#int8-quantization-accuracy-plate-detector)
  below).

### Toolchain

- Hailo AI SW Suite Docker image `hailo8_ai_sw_suite_2025-10:1`
- DFC v3.33.0, Model Zoo v2.17.0, HailoRT v4.23.0 (matched to the target Pi)
- Target arch: **`hailo8`** (the 26-TOPS part — *not* `hailo8l`)
- GPU: NVIDIA Quadro RTX 5000, via NVIDIA Container Toolkit + `--gpus all`
- Pipeline: `.pt → .onnx → .har → .hef`

### Rough steps

1. Prepare calibration images (1024+ clean RGB) on the host, in the
   Docker-mounted directory.
2. Author the model script (`.alls`): normalization, sigmoid output activations,
   CPU NMS postprocess, and the calibration config line.
3. Run the container with GPU passthrough.
4. Compile with `hailomz compile`, forcing the six YOLOv8 end nodes and pointing
   `--model-script` at the custom `.alls`.
5. Retrieve and rename the resulting HEF.

The compile command used:

```bash
hailomz compile yolov8n \
  --ckpt /local/shared_with_docker/plate/license_plate_detector.onnx \
  --calib-path /local/shared_with_docker/plate/calib/ \
  --hw-arch hailo8 \
  --classes 1 \
  --model-script /local/shared_with_docker/plate/yolov8n_plate.alls \
  --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv3.2/cv3.2.2/Conv
```

### Gotchas that cost time

- **Optimization level is forced to 0 by default.** The DFC sets the
  optimization level automatically: level 2 needs a working GPU and ≥1024
  calibration images; CPU-only is always capped at level 0. A level-0 HEF is fine
  for an integration smoke test but not for production accuracy.
- **Path 1 deliberately skips the GPU, so it must be added back for a real
  build.** The official Path 1 `docker run` omits the NVIDIA Container Toolkit and
  `--gpus all`. For a production (level 2) compile you must install the toolkit
  (`nvidia-container-toolkit` + `nvidia-ctk runtime configure --runtime=docker`)
  and recreate the container with `--gpus all`. `--gpus` is a `docker run`
  parameter — it cannot be added to an existing container with `docker start`; the
  container has to be removed and recreated. All persistent work lives on the
  host-mounted volume, so recreating loses nothing.
- **The 64-image calibration default is separate from the directory contents.**
  Even with 1024+ images in `--calib-path`, the DFC defaults to using only 64 of
  them, silently capping the optimization level. The number actually used is
  controlled by `calibset_size` in the `.alls` model script — not by how many
  files are in the directory:

  ```
  model_optimization_config(calibration, batch_size=8, calibset_size=1024)
  ```

- **TensorFlow seeing the GPU ≠ the DFC using it.**
  `tf.config.list_physical_devices('GPU')` returning a device, and even the DFC's
  own system check passing, are not the same thing. The suite image ships with
  CUDA 11.8, and the DFC prints "Recommended" warnings about CUDA 12.5 / cuDNN 9 —
  these are non-fatal and the GPU still gets used. Verify from the compile log
  itself: look for `Using default optimization level of 2`, `Using dataset with
  1024 entries for calibration`, and a fast calibration rate (~20 entries/s = GPU;
  ~3 entries/s = CPU fallback).
- **The dataset's "no augmentation" claim wasn't fully reliable.** The keremberke
  README states no augmentation was applied, but a fraction of images had baked-in
  edge-feathering/matte artifacts (and an earlier Roboflow-derived export of
  similar source imagery had far worse synthetic noise and color-block
  corruption). Always spot-check the actual pixels before calibrating — README
  metadata on re-exported datasets can under-report transforms. Edge-feathering is
  mild and harmless once flattened to RGB on copy (PIL `.convert('RGB')`); the
  synthetic noise/color-block corruption is not, and those images must be excluded.
- **Container UID mismatch on the shared volume.** The container runs as user
  `hailo` (UID 10642, GID 10600). Files the host user creates in the mounted
  directory may not be writable/readable by the container and vice versa. Fix with
  `sudo chown -R 10642:10600 <dir>` on the host for files the container needs, and
  chown back to your user for outputs you need to access from the host.

## INT8 Quantization Accuracy (Plate Detector)

The two INT8 builds above were evaluated for accuracy against the **pristine
validation split** (1,765 images, 1,840 plate instances) of the
[keremberke/license-plate-object-detection](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
dataset — the split deliberately held back during calibration — using
`hailomz eval` on the DFC emulator. A full-precision (FP32) baseline was measured
on the same optimized HAR for comparison.

| Build | Opt level | Calib images | mAP@.50 | mAP@.75 | mAP@.50:.95 | AR@100 |
|-------|:---------:|:------------:|:-------:|:-------:|:-----------:|:------:|
| FP32 (full precision) | — | —    | 0.963 | 0.632 | 0.575 | 0.631 |
| INT8 provisional      | 0 | 64   | 0.963 | 0.627 | 0.570 | 0.625 |
| INT8 production       | 2 | 1024 | 0.959 | 0.662 | 0.589 | 0.647 |

### What the numbers say

**INT8 quantization was effectively lossless for this detector.** All three
builds sit inside a ~2-point mAP band on every metric, and ~0.96 at the
loose IoU = 0.50 threshold.

- The **provisional** build (opt-0, 64 calibration images) lands within 0.5
  mAP points of FP32 across the board — a negligible quantization cost.
- The **production** build (opt-2, 1,024 calibration images) is statistically
  indistinguishable from FP32: marginally above on mAP@.50:.95 / mAP@.75,
  marginally below on mAP@.50. A quantized model cannot genuinely exceed the
  float model it is derived from; small over/under-shoots of this size on a
  1,765-image set reflect calibration-data sampling and quantization noise
  (which can act as mild regularization), not a real accuracy gain.

**Takeaway:** for this single-class detector, INT8 on Hailo-8 preserves
full-precision accuracy, and even a minimal 64-image / opt-0 calibration was
sufficient. The 16x larger calibration set and higher optimization level did
**not** produce a beyond-noise improvement — useful evidence of strongly
diminishing returns on calibration effort for a task this simple.

> Differences of ≤~2 mAP points on a set this size should not be
> over-interpreted without confidence intervals; treat the three builds as
> statistically comparable.

### Method notes (for reproducibility)

- **Emulator, not hardware.** INT8 measured with `--target emulator` (quantized
  numeric emulation on GPU); FP32 measured with `--target full_precision` on the
  same optimized HAR. Neither was run on a physical Hailo-8; on-device HEF
  results may differ slightly.
- **Apples-to-apples.** Every run used the same validation TFRecord, the same
  eval config (single class, `labels_offset: 0`, CPU-NMS baked from the model's
  `.alls`), and `--hw-arch hailo8`. FP32 vs INT8 was measured on the *same*
  optimized HAR, so the only variable is numeric precision. The FP32 result was
  identical across both HARs (they share float weights), confirming consistency.
- **Custom validation TFRecord.** The stock Model Zoo TFRecord builder assumes a
  filename-matching convention between images and labels that does not hold for
  this dataset's layout; a custom builder script was used to pair them correctly.
  Verifying eval behaviour against the v2.17 Model Zoo source — rather than
  trusting the stock builder and the default `labels_offset` — was what surfaced
  this. Both the filename assumption and the label-offset default would otherwise
  have silently zeroed the mAP without ever raising an error.
- **No data leakage.** The validation split was held out from calibration;
  calibration drew only from the train split.
- **Eval commands:**
  ```bash
  # INT8 (per build)
  hailomz eval yolov8n_plate_eval --har <model>.har \
    --data-path lp_val.tfrecord --target emulator --hw-arch hailo8

  # FP32 baseline (same HAR, full precision)
  hailomz eval yolov8n_plate_eval --har license_plate_detector.har \
    --data-path lp_val.tfrecord --target full_precision --hw-arch hailo8
  ```

## License

YOLOv8 models are released under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by
Ultralytics.
