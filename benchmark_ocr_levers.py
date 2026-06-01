#!/usr/bin/env python3
"""
benchmark_ocr_levers.py — measure ANPR OCR-time reduction levers against REAL frames.

Goal: find what reduces license-plate read time WITHOUT changing the decoded plate.
Accuracy is the priority — every lever prints the plate it read so any regression
versus the baseline is visible at a glance, plus an AGREE column.

Levers compared (all on the SAME plate crop per frame):
  baseline        full readtext() on 800px crop        (current production path)
  threads=N       baseline + torch.set_num_threads(N)  (safe: output unchanged)
  width=400       readtext() on a 400px crop           (smaller CRAFT input)
  width+canvas    400px crop + capped canvas_size      (bounds CRAFT upscaling)

The width/canvas levers change the OCR INPUT size, not the recognition logic, so
the decoded plate + an AGREE column are printed for each to catch any regression.

It uses your real Hailo plate detector to localise the plate on each saved
IMX477 frame, exactly as anpr_loop does, so the crops are representative.

PREREQUISITE: stop the detection service first so the Hailo device is free:
    sudo systemctl stop security_camera

USAGE (on the Pi, in the project dir):
    python3 benchmark_ocr_levers.py
    python3 benchmark_ocr_levers.py --images images --max 8 --repeats 2
    python3 benchmark_ocr_levers.py --no-hailo   # skip plate detector, geometric crop only
"""

import argparse
import glob
import os
import sys
import time
import statistics
import re

import cv2
import numpy as np
import yaml


# Matches the trailing timestamp save_frame() appends: _YYYYMMDD_HHMMSS_ffffff.jpg
_TS_RE = re.compile(r"_\d{8}_\d{6}_\d+\.(?:jpg|png|jpeg)$", re.IGNORECASE)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def truth_plate_from_name(path):
    """Recover the plate the system logged at capture time from the filename.

    save_frame() builds 'plate_<PLATE>_<timestamp>.jpg'. We strip the leading
    'plate_' and the trailing timestamp. NOTE: this is the read that passed
    validation when captured, not hand-verified ground truth — spot-check a few.
    Returns an UPPER-cased, single-spaced plate, or '' if it can't be parsed.
    """
    base = os.path.basename(path)
    if not base.lower().startswith("plate_"):
        return ""
    stripped = _TS_RE.sub("", base)          # remove trailing timestamp+ext
    plate = stripped[len("plate_"):]          # remove leading 'plate_'
    plate = re.sub(r"[^A-Z0-9 ]", "", plate.upper())
    return " ".join(plate.split())


def find_frames(images_dir, limit, spread=False):
    """Saved ANPR captures are named plate_*.jpg by save_frame().

    spread=True samples evenly across the (time-sorted) set instead of taking
    the first `limit`, so you get variety across dates/times rather than a run
    of near-identical consecutive captures.
    """
    paths = sorted(glob.glob(os.path.join(images_dir, "plate_*.jpg")))
    if not paths:
        # fall back to any jpg/png so the script is still useful pre-deployment
        paths = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.png"))
        )
    if not limit or len(paths) <= limit:
        return paths
    if spread:
        step = len(paths) / limit
        return [paths[int(i * step)] for i in range(limit)]
    return paths[:limit]


def crop_from_plate_bbox(image, plate_bbox):
    """Mirror anpr_module.read_plate's tight-crop logic (with padding)."""
    img_h, img_w = image.shape[:2]
    px1, py1, px2, py2 = plate_bbox
    pw, ph = px2 - px1, py2 - py1
    pad_x, pad_y = int(pw * 0.15), int(ph * 0.25)
    x1 = max(0, px1 - pad_x)
    y1 = max(0, py1 - pad_y)
    x2 = min(img_w, px2 + pad_x)
    y2 = min(img_h, py2 + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def resize_max_width(image, max_w):
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / w
    return cv2.resize(image, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def best_plate_text(ocr_results):
    """Pick the highest-confidence non-trivial token. Deliberately simple:
    this benchmark measures speed + read-stability, not the full validation/
    combination logic in anpr_module (which is identical across all levers,
    so it cannot change the RELATIVE comparison)."""
    best_txt, best_conf = "", 0.0
    for r in ocr_results:
        txt = "".join(c for c in r[1].upper() if c.isalnum() or c == " ").strip()
        conf = float(r[2])
        if len(txt) >= 4 and conf > best_conf:
            best_txt, best_conf = txt, conf
    return best_txt, best_conf


def time_call(fn, repeats):
    """Run fn() `repeats` times, return (median_seconds, last_result).
    First call per lever is a warm-up and is discarded when repeats > 1."""
    times, result = [], None
    runs = repeats + 1 if repeats > 1 else 1
    for i in range(runs):
        t0 = time.time()
        result = fn()
        dt = time.time() - t0
        if not (repeats > 1 and i == 0):  # drop warm-up
            times.append(dt)
    return statistics.median(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dir of saved frames (default: config storage.image_path)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max", type=int, default=12, help="max frames to test")
    ap.add_argument("--repeats", type=int, default=2, help="timed repeats per lever (median reported, warm-up dropped)")
    ap.add_argument("--no-hailo", action="store_true", help="skip plate detector; geometric-only crop")
    ap.add_argument("--thread-counts", default="2,3,4", help="comma list of torch thread counts to test")
    ap.add_argument("--no-spread", action="store_true",
                    help="take the first N frames instead of sampling evenly across all (default: spread)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    images_dir = args.images or cfg.get("storage", {}).get("image_path", "images")
    frames = find_frames(images_dir, args.max, spread=not args.no_spread)
    if not frames:
        print(f"No frames found in '{images_dir}'. Run the system to capture some plate_*.jpg first.")
        sys.exit(1)

    sel = "first" if args.no_spread else "spread across all captures"
    print(f"Frames: {len(frames)} from '{images_dir}' ({sel})  | repeats={args.repeats} (warm-up dropped)\n")

    import torch
    import easyocr

    # --- Readers (loaded once) ---
    print("Loading EasyOCR reader (this is one-time, not part of per-read timing)...")
    reader_full = easyocr.Reader(["en"], gpu=False)            # detection + recognition
    print("Reader ready.\n")

    # --- Hailo plate detector (optional) ---
    plate_detector = None
    if not args.no_hailo:
        try:
            from hailo_detector import HailoDetector
            det = cfg["detection"]
            vdev = HailoDetector.create_shared_vdevice()
            plate_detector = HailoDetector(
                det["plate_model_path"],
                confidence_threshold=det.get("plate_confidence_threshold", 0.25),
                class_names=["license_plate"],
                vdevice=vdev,
            )
            print("Hailo plate detector loaded (shared VDevice).\n")
        except Exception as e:
            print(f"Could not load Hailo plate detector ({e}). Continuing geometric-only.\n")
            plate_detector = None

    thread_counts = [int(x) for x in args.thread_counts.split(",") if x.strip()]

    # Accumulators: lever -> list of (seconds, read_text, truth_text)
    results = {}

    def record(lever, seconds, text, truth):
        results.setdefault(lever, []).append((seconds, text, truth))

    for fpath in frames:
        image = cv2.imread(fpath)  # BGR, matches capture path on Trixie
        if image is None:
            print(f"  skip (unreadable): {fpath}")
            continue
        name = os.path.basename(fpath)
        truth = truth_plate_from_name(fpath)
        print(f"=== {name}  ({image.shape[1]}x{image.shape[0]}) ===")
        print(f"  truth (from filename): '{truth}'")

        # Localise the plate with Hailo, as anpr_loop does
        plate_bbox, plate_conf = None, 0.0
        if plate_detector is not None:
            dets = plate_detector.detect(image)
            if dets:
                best = max(dets, key=lambda d: d["confidence"])
                plate_bbox = best["bbox"]
                plate_conf = best["confidence"]
                print(f"  plate localised: bbox={plate_bbox} conf={plate_conf:.2f}")
            else:
                print("  plate detector: NO plate found")

        crop = crop_from_plate_bbox(image, plate_bbox) if plate_bbox else None
        if crop is None:
            print("  (no tight crop available; skipping frame for lever comparison)\n")
            continue

        crop_full = resize_max_width(crop, 800)   # baseline width cap (matches anpr_module)
        crop_tight = resize_max_width(crop, 400)  # tighter cap for width/canvas levers

        def tag(txt):
            return "TRUE" if truth and txt == truth else ("----" if truth else "?")

        # 1) baseline
        torch.set_num_threads(torch.get_num_threads())  # leave as default for baseline
        sec, res = time_call(lambda: reader_full.readtext(crop_full), args.repeats)
        base_txt, base_conf = best_plate_text(res)
        record("baseline", sec, base_txt, truth)
        print(f"  baseline        {sec:5.2f}s  read='{base_txt}' ({base_conf:.2f})  [{tag(base_txt)}]")

        # 2) thread-count sweep (output should be identical to baseline)
        for n in thread_counts:
            torch.set_num_threads(n)
            sec, res = time_call(lambda: reader_full.readtext(crop_full), args.repeats)
            txt, conf = best_plate_text(res)
            record(f"threads={n}", sec, txt, truth)
            b = "=base" if txt == base_txt else "DIFF "
            print(f"  threads={n}       {sec:5.2f}s  read='{txt}' ({conf:.2f})  [{b} {tag(txt)}]")
        torch.set_num_threads(thread_counts[0] if thread_counts else 4)

        # 3a) width=600: middle ground between 800 (accurate/slow) and 400 (fast).
        crop_600 = resize_max_width(crop, 600)
        sec, res = time_call(lambda: reader_full.readtext(crop_600), args.repeats)
        txt, conf = best_plate_text(res)
        record("width=600", sec, txt, truth)
        b = "=base" if txt == base_txt else "DIFF "
        print(f"  width=600       {sec:5.2f}s  read='{txt}' ({conf:.2f})  [{b} {tag(txt)}]")

        # 3b) width=400 only: smaller crop, default canvas_size/mag_ratio.
        sec, res = time_call(lambda: reader_full.readtext(crop_tight), args.repeats)
        txt, conf = best_plate_text(res)
        record("width=400", sec, txt, truth)
        b = "=base" if txt == base_txt else "DIFF "
        print(f"  width=400       {sec:5.2f}s  read='{txt}' ({conf:.2f})  [{b} {tag(txt)}]")

        # 3c) width=400 + capped CRAFT canvas_size + mag_ratio=1.0.
        sec, res = time_call(
            lambda: reader_full.readtext(crop_tight, canvas_size=1280, mag_ratio=1.0),
            args.repeats,
        )
        txt, conf = best_plate_text(res)
        record("width+canvas", sec, txt, truth)
        b = "=base" if txt == base_txt else "DIFF "
        print(f"  width+canvas    {sec:5.2f}s  read='{txt}' ({conf:.2f})  [{b} {tag(txt)}]")
        print()

    # --- Summary ---
    if not results:
        print("No comparable frames (no tight crops produced). "
              "Capture some plate_*.jpg with a localisable plate first.")
        return

    base_reads = {i: t for i, (_, t, _) in enumerate(results.get("baseline", []))}
    # How many frames even have a usable filename-truth to score against?
    truth_rows = [tr for (_, _, tr) in results["baseline"] if tr]
    n_truth = len(truth_rows)

    print("=" * 72)
    print(f"{'lever':<16}{'median s':>10}{'vs base':>9}{'correct%':>10}{'=base%':>9}")
    print("-" * 72)
    base_median = statistics.median([s for s, _, _ in results["baseline"]])
    for lever, rows in results.items():
        med = statistics.median([s for s, _, _ in rows])
        speedup = base_median / med if med else 0
        # correct% = matches filename-truth, over frames that HAVE a truth
        correct_n = sum(1 for (_, txt, tr) in rows if tr and txt == tr)
        correct_pct = (100.0 * correct_n / n_truth) if n_truth else float("nan")
        # =base% = matches the baseline read (the old metric), over all frames
        base_n = sum(1 for i, (_, txt, _) in enumerate(rows) if base_reads.get(i) == txt)
        base_pct = 100.0 * base_n / len(rows) if rows else 0
        sp = "" if lever == "baseline" else f"{speedup:4.2f}x"
        cp = "  n/a" if n_truth == 0 else f"{correct_pct:4.0f}%"
        print(f"{lever:<16}{med:>10.2f}{sp:>9}{cp:>10}{base_pct:>8.0f}%")
    print("=" * 72)
    print(f"\nScored against filename-truth on {n_truth}/{len(results['baseline'])} frames "
          f"(frames whose name encodes a plate).")
    print("Read correct% FIRST — it scores against the logged plate, not the 800px\n"
          "baseline (which can itself be wrong). =base% is the old 'matches baseline'\n"
          "metric, kept for continuity. A lever wins if it is FASTER and its correct%\n"
          "is >= baseline's. If baseline's own correct% is low, the filename-truth may\n"
          "be unreliable — spot-check a few frames by eye before trusting the table.")


if __name__ == "__main__":
    main()
