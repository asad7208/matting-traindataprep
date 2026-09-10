#!/usr/bin/env bash
# Dump evenly-spaced frames out of a video.
#
# EASIEST WAY: fill in VIDEO and OUTPUT below, then run
#   ./run_extract_frames.sh
# It prints the video's fps / total frames / duration, then asks how many
# frames you want.
#
# The command line still wins over the variables:
#   ./run_extract_frames.sh clip.mp4 /data/out 200
set -euo pipefail

# =========================== EDIT THESE ==================================
# VIDEO: the video file to pull frames from.
VIDEO="data/CCPL080626M1_1_16_1/camera04.mp4"

# OUTPUT: frames land in OUTPUT/<video name>/, so several videos can share it.
OUTPUT="frames"

# NUM_FRAMES: leave empty to be asked after the video info is printed.
# Put a number here (or "all") to skip the question entirely.
NUM_FRAMES=""

# Image format for the dumped frames.
EXT="jpg"          # jpg or png
QUALITY="95"       # jpg only, 1-100
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Positional arguments override the variables: <video> [output] [num_frames]
[ $# -ge 1 ] && VIDEO="$1"
[ $# -ge 2 ] && OUTPUT="$2"
[ $# -ge 3 ] && NUM_FRAMES="$3"

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

ARGS=(--output "$OUTPUT" --ext "$EXT" --quality "$QUALITY")
[ -n "$NUM_FRAMES" ] && ARGS+=(--num-frames "$NUM_FRAMES")

exec "$PY" "$SCRIPT_DIR/extract_frames.py" "$VIDEO" "${ARGS[@]}"
