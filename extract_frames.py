#!/usr/bin/env python3
"""Extract N evenly-spaced frames from a video into <output>/<video name>/."""

import argparse
import os
import sys
import time

import cv2


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024


def parse_count(raw, total):
    """A frame count, 'all', or None when it is not a usable number."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("all", "max", "everything"):
        return total
    try:
        n = int(text)
    except ValueError:
        return None
    return n if n >= 1 else None


def suggestions(total, fps):
    """A few sensible counts, described in seconds so they mean something."""
    out = []
    if fps > 0:
        for per_sec, label in ((1, "1 frame per second"),
                               (0.5, "1 frame every 2 seconds"),
                               (0.2, "1 frame every 5 seconds")):
            n = int(round(total / fps * per_sec))
            if 1 <= n <= total:
                out.append((label, n))
    for n in (100, 300, 1000):
        if n < total:
            out.append((f"a flat {n}", n))
    out.append(("every frame", total))
    seen, uniq = set(), []
    for label, n in out:
        if n not in seen:
            seen.add(n)
            uniq.append((label, n))
    return uniq


def bar(done, want, t0, width=30):
    frac = min(1.0, done / want) if want else 1.0
    fill = int(round(frac * width))
    el = time.time() - t0
    eta = (el / frac - el) if frac > 0.02 else 0
    clock = lambda s: f"{int(s) // 60:d}:{int(s) % 60:02d}"       # noqa: E731
    print(f"\r  [{'#' * fill}{'-' * (width - fill)}] {frac * 100:5.1f}%  "
          f"{done}/{want}  {clock(el)}" + (f"<{clock(eta)}" if eta else ""),
          end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to input video")
    ap.add_argument("-o", "--output", default="frames", help="output location (default: ./frames)")
    ap.add_argument("-n", "--num-frames", default=None,
                    help="how many frames to extract, or 'all' (asked "
                         "interactively if omitted)")
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

    mins, secs = divmod(duration, 60)
    print("-" * 58)
    print(f"  video        : {args.video}")
    print(f"  file size    : {human_size(file_size)}")
    print(f"  resolution   : {width} x {height}")
    print(f"  fps          : {fps:.3f}")
    print(f"  total frames : {total}")
    print(f"  duration     : {duration:.2f} s  ({int(mins)}m {secs:04.1f}s)")
    print("-" * 58)

    if total <= 0:
        sys.exit("error: could not determine frame count for this video")

    want = parse_count(args.num_frames, total)
    if want is None:
        # Suggestions in the video's own terms, so the number means something.
        print("  suggestions:")
        for label, n in suggestions(total, fps):
            print(f"    {n:>6}  {label}")
        while want is None:
            try:
                raw = input(f"\n  how many frames? (1-{total}, or 'all') "
                            f"[{min(300, total)}]: ").strip()
            except EOFError:
                sys.exit("\nerror: no --num-frames given and no answer to "
                         "read. Set NUM_FRAMES in run_extract_frames.sh.")
            want = parse_count(raw or str(min(300, total)), total)
            if want is None:
                print("    give a whole number, or 'all'")

    if want > total:
        print(f"  note: asked for {want} but the video has {total}, using {total}")
        want = total

    step = max(1, total // want)
    every = step / fps if fps > 0 else 0
    print(f"  taking every {step} frame(s)"
          + (f" — about one every {every:.2f}s of video" if every else ""))

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
    t0 = time.time()
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
            bar(saved, want, t0)
        idx += 1

    bar(saved, want, t0)
    cap.release()
    print(f"\n  done: {saved} frame(s) -> {out_dir}")


if __name__ == "__main__":
    main()
