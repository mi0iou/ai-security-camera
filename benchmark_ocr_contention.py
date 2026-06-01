#!/usr/bin/env python3
"""
benchmark_ocr_contention.py — find the OCR settings that minimise LIVE plate-read
time without losing accuracy, measured UNDER REAL CONTENTION.

Why this exists (what the earlier isolated benchmark could not show):
  - Live reads cost ~9s; the same crop costs ~2.2s in isolation. The 4x gap is
    CPU contention between EasyOCR (PyTorch) and the running detection loop.
  - The "tight" plate crop is often NOT small: a close plate is ~1000px wide, so
    after the old fixed 800px cap EasyOCR still chews ~800x340 — no pixel saving
    over the geometric crop. Plate size varies a lot with vehicle distance, so a
    fixed crop width is the wrong lever.

What this does differently:
  1. RUNS WITH THE DETECTION SERVICE LIVE — real contention, the target regime.
  2. Reproduces the exact live crop from <frame>.json sidecars written by main.py
     (no re-localisation, so NO extra Hailo/NPU contention from this benchmark).
  3. Sweeps PLATE-HEIGHT-NORMALISED targets, not fixed widths: resize so the
     plate's height lands at a target, scaling correctly for near vs far plates.
  4. Sweeps torch thread counts (contention is confirmed as the multiplier).
  5. Scores correct% against the plate in the filename (the logged read), so a
     faster setting is only a win if it still reads every plate-distance right.

PREREQUISITE: leave the detection service RUNNING (that is the whole point):
    # do NOT stop security_camera
Capture some frames first by letting vehicles pass; main.py writes
plate_<NUM>_<ts>.jpg + plate_<NUM>_<ts>.json pairs.

USAGE (on the Pi, in the project dir, service running):
    python3 benchmark_ocr_contention.py
    python3 benchmark_ocr_contention.py --plate-heights 32,48,64,96 --thread-counts 1,2,3
    python3 benchmark_ocr_contention.py --max 20

NOTE: This benchmark creates its OWN EasyOCR Reader (CPU). It does NOT touch the
Hailo device, so it is safe to run alongside the live service. Expect the live
detection FPS to dip while this runs — that dip is the contention you're measuring.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

import cv2
import numpy as np
import yaml

_TS_RE = re.compile(r"_\d{8}_\d{6}_\d+\.(?:jpg|png|jpeg)$", re.IGNORECASE)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def truth_plate_from_name(path):
    base = os.path.basename(path)
    if not base.lower().startswith("plate_"):
        return ""
    stripped = _TS_RE.sub("", base)
    plate = stripped[len("plate_"):]
    plate = re.sub(r"[^A-Z0-9 ]", "", plate.upper())
    return " ".join(plate.split())


def find_pairs(images_dir, limit, spread=True):
    """Return [(jpg_path, json_path, meta), ...] for frames that have a sidecar
    with a usable plate_bbox."""
    jpgs = sorted(glob.glob(os.path.join(images_dir, "plate_*.jpg")))
    pairs = []
    for j in jpgs:
        side = os.path.splitext(j)[0] + ".json"
        if not os.path.exists(side):
            continue
        try:
            with open(side) as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("plate_bbox"):
            pairs.append((j, side, meta))
    if not pairs:
        return pairs
    if limit and len(pairs) > limit:
        if spread:
            step = len(pairs) / limit
            pairs = [pairs[int(i * step)] for i in range(limit)]
        else:
            pairs = pairs[:limit]
    return pairs


def crop_from_bbox(image, bbox):
    """Mirror anpr_module.read_plate's tight-crop padding (15% x / 25% y)."""
    img_h, img_w = image.shape[:2]
    px1, py1, px2, py2 = bbox
    pw, ph = px2 - px1, py2 - py1
    pad_x, pad_y = int(pw * 0.15), int(ph * 0.25)
    x1 = max(0, px1 - pad_x)
    y1 = max(0, py1 - pad_y)
    x2 = min(img_w, px2 + pad_x)
    y2 = min(img_h, py2 + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def resize_to_plate_height(crop, plate_px_h, max_h):
    """Cap the crop so the PLATE height is AT MOST max_h px. This only ever
    DOWNSCALES: a plate already shorter than max_h (a distant vehicle) is passed
    through untouched, preserving its limited detail. A large close plate is
    shrunk to the cap, which is where the live speed-up comes from.

    This is the key correction from the failed 'normalise to a fixed target'
    approach: distant plates must not be forced up OR down to a target, only
    oversized plates get capped.
    """
    if plate_px_h <= 0 or plate_px_h <= max_h:
        return crop  # already at/under the cap -> leave it alone
    scale = max_h / plate_px_h  # < 1.0, downscale only
    h, w = crop.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def best_plate_text(ocr_results):
    best_txt, best_conf = "", 0.0
    for r in ocr_results:
        txt = "".join(c for c in r[1].upper() if c.isalnum() or c == " ").strip()
        conf = float(r[2])
        if len(txt) >= 4 and conf > best_conf:
            best_txt, best_conf = txt, conf
    return best_txt, best_conf


def time_call(fn, repeats):
    times, result = [], None
    runs = repeats + 1 if repeats > 1 else 1
    for i in range(runs):
        t0 = time.time()
        result = fn()
        dt = time.time() - t0
        if not (repeats > 1 and i == 0):
            times.append(dt)
    return statistics.median(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dir of saved frames (default: config storage.image_path)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max", type=int, default=12, help="max frame+sidecar pairs to test")
    ap.add_argument("--repeats", type=int, default=2, help="timed repeats per setting (median; warm-up dropped)")
    ap.add_argument("--plate-heights", default="96,128,160,800",
                    help="comma list of MAX plate heights in px (a ceiling, not a target). "
                         "A plate already shorter is left untouched. Large value (800) = "
                         "effectively 'no cap' baseline / accuracy floor.")
    ap.add_argument("--thread-counts", default="1,2,4",
                    help="comma list of torch thread counts to sweep")
    ap.add_argument("--no-spread", action="store_true", help="first N pairs instead of evenly sampled")
    args = ap.parse_args()

    cfg = load_config(args.config)
    images_dir = args.images or cfg.get("storage", {}).get("image_path", "images")
    pairs = find_pairs(images_dir, args.max, spread=not args.no_spread)
    if not pairs:
        print(f"No frame+sidecar pairs in '{images_dir}'.")
        print("Make sure main.py with the sidecar writer has captured some plate_*.json")
        print("files (let a few vehicles pass with the service running), then retry.")
        sys.exit(1)

    plate_heights = [int(x) for x in args.plate_heights.split(",") if x.strip()]
    thread_counts = [int(x) for x in args.thread_counts.split(",") if x.strip()]

    sel = "first" if args.no_spread else "spread across captures"
    print(f"Pairs: {len(pairs)} from '{images_dir}' ({sel})")
    print(f"Target plate heights: {plate_heights}px   Thread counts: {thread_counts}")
    print("Detection service should be RUNNING — this measures live contention.\n")

    import torch
    import easyocr
    print("Loading EasyOCR reader (one-time, not timed)...")
    reader = easyocr.Reader(["en"], gpu=False)
    print("Reader ready.\n")

    # results[(plate_h, threads)] = list of (seconds, read_text, truth, in_w, in_h)
    results = {}

    for jpg, side, meta in pairs:
        image = cv2.imread(jpg)
        if image is None:
            print(f"  skip (unreadable): {jpg}")
            continue
        truth = truth_plate_from_name(jpg)
        bbox = meta["plate_bbox"]
        crop = crop_from_bbox(image, bbox)
        if crop is None:
            print(f"  skip (empty crop): {os.path.basename(jpg)}")
            continue
        plate_px_h = bbox[3] - bbox[1]  # plate height in full-frame px
        live_t = meta.get("ocr_conf")
        print(f"=== {os.path.basename(jpg)}  truth='{truth}'  "
              f"plate_h={plate_px_h}px  crop={crop.shape[1]}x{crop.shape[0]} ===")

        for ph in plate_heights:
            ocr_img = resize_to_plate_height(crop, plate_px_h, ph)
            in_w, in_h = ocr_img.shape[1], ocr_img.shape[0]
            for nt in thread_counts:
                torch.set_num_threads(nt)
                sec, res = time_call(lambda: reader.readtext(ocr_img), args.repeats)
                txt, conf = best_plate_text(res)
                key = (ph, nt)
                results.setdefault(key, []).append((sec, txt, truth, in_w, in_h))
                ok = "TRUE" if (truth and txt == truth) else "----"
                capped = "cap" if plate_px_h > ph else "as-is"
                print(f"   max_h={ph:>4} threads={nt}  {sec:5.2f}s  "
                      f"in={in_w}x{in_h} {capped}  read='{txt}' ({conf:.2f}) [{ok}]")
        print()

    # ---- Summary ----
    if not results:
        print("No comparable pairs produced. Capture more frames and retry.")
        return

    n_truth = sum(1 for (_, _, tr, _, _) in next(iter(results.values())) if tr)
    print("=" * 74)
    print(f"{'max_h':>8}{'threads':>9}{'median s':>11}{'correct%':>11}{'avg in_w':>11}")
    print("-" * 74)
    # sort by plate_h then threads for readability
    for key in sorted(results.keys()):
        ph, nt = key
        rows = results[key]
        med = statistics.median([s for s, _, _, _, _ in rows])
        correct = sum(1 for (_, txt, tr, _, _) in rows if tr and txt == tr)
        correct_pct = (100.0 * correct / n_truth) if n_truth else float("nan")
        avg_w = statistics.mean([w for _, _, _, w, _ in rows])
        cp = "  n/a" if n_truth == 0 else f"{correct_pct:4.0f}%"
        print(f"{ph:>8}{nt:>9}{med:>11.2f}{cp:>11}{avg_w:>11.0f}")
    print("=" * 74)
    print(f"\nScored against filename-truth on {n_truth} frames.")
    print("max_h is a CEILING: plates already shorter pass through untouched, so a\n"
          "low cap cannot hurt distant plates (it never fires on them) — it only\n"
          "shrinks oversized close plates. Pick the LOWEST max_h that still holds\n"
          "100% correct% (== the 800 baseline). That is the safe ceiling to put in\n"
          "anpr_module._resize_for_ocr, paired with torch.set_num_threads(1).")


if __name__ == "__main__":
    main()
