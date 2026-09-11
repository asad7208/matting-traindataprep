#!/usr/bin/env python3
"""Detect people in a video, pad each bbox, and save the crops grouped by size.

Every crop goes to <output>/<video name>/<size bucket>/, where the bucket comes
from the crop's total pixel count (width * height) after padding.
"""

import argparse
import json
import os
import sys
import time

import cv2

# (folder name, exclusive upper bound on width * height). Prefixed with a number
# so the buckets list in size order.
BUCKETS = [
    ("01_tiny_lt_64x64",      64 * 64),
    ("02_small_lt_128x128",   128 * 128),
    ("03_medium_lt_256x256",  256 * 256),
    ("04_large_lt_512x512",   512 * 512),
    ("05_xlarge_lt_1024x1024", 1024 * 1024),
    ("06_huge_ge_1024x1024",  float("inf")),
]


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024


def bucket_for(area):
    for name, limit in BUCKETS:
        if area < limit:
            return name
    return BUCKETS[-1][0]


def pad_box(x1, y1, x2, y2, pad, w, h):
    """Grow the box by `pad` (a fraction of its own size) and clamp to frame."""
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * pad, bh * pad
    x1 = max(0, int(round(x1 - dx)))
    y1 = max(0, int(round(y1 - dy)))
    x2 = min(w, int(round(x2 + dx)))
    y2 = min(h, int(round(y2 + dy)))
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to input video")
    ap.add_argument("-o", "--output", default="crops",
                    help="output location (default: ./crops)")
    ap.add_argument("--model", default="weights/yolo11x.pt",
                    help="YOLO detection weights (default: weights/yolo11x.pt)")
    ap.add_argument("--pad", type=float, default=0.25,
                    help="grow each bbox by this fraction per side (default: 0.25)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="detection confidence threshold (default: 0.35)")
    ap.add_argument("--stride", type=int, default=1,
                    help="only run detection on every Nth frame (default: 1)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed frames (0 = whole video)")
    ap.add_argument("--min-pixels", type=int, default=0,
                    help="skip crops smaller than this many pixels (default: 0)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="detector input size (default: 1280)")
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. 0 or cpu (default: ultralytics picks)")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"],
                    help="image format (default: jpg)")
    ap.add_argument("--quality", type=int, default=95,
                    help="jpg quality 1-100 (default: 95)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"error: no such file: {args.video}")
    if not os.path.isfile(args.model):
        sys.exit(f"error: no such weights file: {args.model}")

    from tqdm import tqdm
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"error: could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0
    name = os.path.splitext(os.path.basename(args.video))[0]

    mins, secs = divmod(duration, 60)
    print("-" * 58)
    print(f"  video        : {args.video}")
    print(f"  file size    : {human_size(os.path.getsize(args.video))}")
    print(f"  resolution   : {width} x {height}")
    print(f"  fps          : {fps:.3f}")
    print(f"  total frames : {total}")
    print(f"  duration     : {duration:.2f} s  ({int(mins)}m {secs:04.1f}s)")
    print(f"  model        : {args.model}")
    print(f"  bbox padding : {args.pad * 100:.0f}% per side")
    print("-" * 58)

    stride = max(1, args.stride)
    to_process = (total + stride - 1) // stride if total > 0 else 0
    if args.max_frames > 0:
        to_process = min(to_process, args.max_frames) if to_process else args.max_frames

    model = YOLO(args.model)
    out_root = os.path.join(args.output, name)
    os.makedirs(out_root, exist_ok=True)

    if args.ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.quality))]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    pad_width = max(6, len(str(total)))
    counts = {folder: 0 for folder, _ in BUCKETS}
    records = []
    saved = skipped = processed = 0
    idx = 0
    t0 = time.time()

    progress = tqdm(total=to_process or None, unit="frame", desc="  cropping",
                    dynamic_ncols=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue

            res = model.predict(frame, classes=[0], conf=args.conf,
                                imgsz=args.imgsz, device=args.device,
                                verbose=False)[0]
            fh, fw = frame.shape[:2]
            for det, box in enumerate(res.boxes):
                x1, y1, x2, y2 = pad_box(*box.xyxy[0].tolist(), args.pad, fw, fh)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < args.min_pixels:
                    skipped += 1
                    continue
                folder = bucket_for(area)
                dest = os.path.join(out_root, folder)
                os.makedirs(dest, exist_ok=True)
                fname = f"{name}_{idx:0{pad_width}d}_p{det:02d}.{args.ext}"
                path = os.path.join(dest, fname)
                if cv2.imwrite(path, frame[y1:y2, x1:x2], params):
                    counts[folder] += 1
                    saved += 1
                    records.append({
                        "file": os.path.join(folder, fname),
                        "frame": idx,
                        "bbox_padded": [x1, y1, x2, y2],
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "pixels": area,
                        "bucket": folder,
                        "conf": round(float(box.conf[0]), 4),
                    })
                else:
                    print(f"\nwarn: failed to write {path}")

            processed += 1
            idx += 1
            progress.update(1)
            progress.set_postfix(crops=saved, refresh=False)
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        progress.close()
        cap.release()

    manifest = {
        "video": os.path.abspath(args.video),
        "model": args.model,
        "pad": args.pad,
        "conf": args.conf,
        "stride": stride,
        "frames_processed": processed,
        "crops": saved,
        "buckets": counts,
        "items": records,
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n  frames processed : {processed}")
    print(f"  crops saved      : {saved}"
          + (f"   (skipped {skipped} under --min-pixels)" if skipped else ""))
    for folder, _ in BUCKETS:
        if counts[folder]:
            print(f"    {counts[folder]:>7}  {folder}")
    print(f"  elapsed          : {time.time() - t0:.1f}s")
    print(f"  done -> {out_root}")


if __name__ == "__main__":
    main()
