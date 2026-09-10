#!/usr/bin/env bash
# Usage:
#   ./run_extract_frames.sh <video> [output_dir] [num_frames]
#
# Examples:
#   ./run_extract_frames.sh clip.mp4                     # asks how many frames, saves to ./frames/clip/
#   ./run_extract_frames.sh clip.mp4 /data/out           # asks how many frames
#   ./run_extract_frames.sh clip.mp4 /data/out 200       # non-interactive, 200 frames
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <video> [output_dir] [num_frames]" >&2
    exit 1
fi

VIDEO="$1"
OUTPUT="${2:-frames}"
NUM="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

if [ -n "$NUM" ]; then
    "$PY" "$SCRIPT_DIR/extract_frames.py" "$VIDEO" -o "$OUTPUT" -n "$NUM"
else
    "$PY" "$SCRIPT_DIR/extract_frames.py" "$VIDEO" -o "$OUTPUT"
fi
