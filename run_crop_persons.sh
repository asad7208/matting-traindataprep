#!/usr/bin/env bash
# Detect people in a video, pad each bbox, and save the crops grouped by size.
#
# EASIEST WAY: fill in VIDEO and OUTPUT below, then run
#   ./run_crop_persons.sh
#
# The command line still wins over the variables:
#   ./run_crop_persons.sh clip.mp4 /data/crops 0.25
set -euo pipefail

# =========================== EDIT THESE ==================================
# VIDEO: the video file to pull person crops from.
VIDEO="data/CAM1.mp4"

# OUTPUT: crops land in OUTPUT/<video name>/<size bucket>/, so several videos
# can share one output folder. Buckets are by total pixel count of the crop:
#   01_tiny_lt_64x64  02_small_lt_128x128  03_medium_lt_256x256
#   04_large_lt_512x512  05_xlarge_lt_1024x1024  06_huge_ge_1024x1024
OUTPUT="crops"

# PAD: grow each detected bbox by this fraction on every side. 0.25 = 25%.
PAD="0.25"

# Detector settings.
MODEL="weights/yolo26x.pt"
CONF="0.35"        # confidence threshold
IMGSZ="1280"       # detector input size
DEVICE="0"          # "0" for the first GPU, "cpu" to force CPU, empty = auto

# STRIDE: run detection on every Nth frame (1 = every frame).
STRIDE="1"

# MAX_FRAMES: stop after this many processed frames (0 or empty = whole video).
MAX_FRAMES=""

# MIN_PIXELS: drop crops smaller than this many pixels (0 = keep everything).
MIN_PIXELS="0"

# Image format for the saved crops.
EXT="jpg"          # jpg or png
QUALITY="95"       # jpg only, 1-100
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Positional arguments override the variables: <video> [output] [pad]
[ $# -ge 1 ] && VIDEO="$1"
[ $# -ge 2 ] && OUTPUT="$2"
[ $# -ge 3 ] && PAD="$3"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
        PY="$SCRIPT_DIR/.venv/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PY="$CONDA_PREFIX/bin/python"
    else
        PY="python3"
    fi
fi

if [ -z "$VIDEO" ]; then
    echo "error: set VIDEO at the top of this script, or pass one" >&2
    exit 1
fi
if [ ! -f "$VIDEO" ]; then
    echo "error: no such video: $VIDEO" >&2
    exit 1
fi

ARGS=(--output "$OUTPUT" --model "$MODEL" --pad "$PAD" --conf "$CONF"
      --imgsz "$IMGSZ" --stride "$STRIDE" --min-pixels "$MIN_PIXELS"
      --ext "$EXT" --quality "$QUALITY")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
[ -n "$MAX_FRAMES" ] && ARGS+=(--max-frames "$MAX_FRAMES")

exec "$PY" "$SCRIPT_DIR/crop_persons.py" "$VIDEO" "${ARGS[@]}"
