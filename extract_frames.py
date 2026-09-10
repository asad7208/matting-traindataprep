#!/usr/bin/env python3
"""Extract N evenly-spaced frames from a video into <output>/<video name>/."""

import argparse
import os
import sys

import cv2


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to input video")
    ap.add_argument("-o", "--output", default="frames", help="output location (default: ./frames)")
    ap.add_argument("-n", "--num-frames", type=int, default=None,
                    help="how many frames to extract (asked interactively if omitted)")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="image format (default: jpg)")
    ap.add_argument("--quality", type=int, default=95, help="jpg quality 1-100 (default: 95)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"error: no such file: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"error: could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file_size = os.path.getsize(args.video)
    duration = total / fps if fps > 0 else 0.0

    name = os.path.splitext(os.path.basename(args.video))[0]

    print("-" * 50)
    print(f"video      : {args.video}")
    print(f"file size  : {human_size(file_size)}")
    print(f"resolution : {width} x {height}")
    print(f"fps        : {fps:.3f}")
    print(f"total frames: {total}")
    print(f"duration   : {duration:.2f} s")
    print("-" * 50)

    if total <= 0:
        sys.exit("error: could not determine frame count for this video")

    want = args.num_frames
    if want is None:
        try:
            want = int(input(f"how many frames do you want? (1-{total}): ").strip())
        except (ValueError, EOFError):
            sys.exit("error: invalid number")

    if want < 1:
        sys.exit("error: number of frames must be >= 1")
    if want > total:
        print(f"note: requested {want} > total {total}, using {total}")
        want = total

    step = max(1, total // want)
    print(f"skipping every {step} frame(s) -> ~{len(range(0, total, step)[:want])} frames")

    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)

    if args.ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.quality))]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    pad = max(6, len(str(total)))
    saved = 0
    idx = 0
    next_wanted = 0
    while saved < want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx == next_wanted:
            path = os.path.join(out_dir, f"{name}_{idx:0{pad}d}.{args.ext}")
            if cv2.imwrite(path, frame, params):
                saved += 1
            else:
                print(f"warn: failed to write {path}")
            next_wanted += step
            print(f"\rsaved {saved}/{want}", end="", flush=True)
        idx += 1

    cap.release()
    print(f"\ndone: {saved} frames -> {out_dir}")


if __name__ == "__main__":
    main()
