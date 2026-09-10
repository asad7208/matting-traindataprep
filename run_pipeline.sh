#!/usr/bin/env bash
# One-shot terminal pipeline: mine (track/pose/CLIP/dedupe/select) -> review ->
# crops -> BiRefNet mattes -> final cutout images.
#
# EASIEST WAY: fill in SOURCE and OUT just below, then run
#   ./run_pipeline.sh
#
# Anything on the command line still wins over the variables:
#   ./run_pipeline.sh --source clip.mp4 --out ds --budget 200 --final-mode green -y
#
# Only re-matte an existing crops folder:
#   ./run_pipeline.sh --stages masks,final -y
#
# Every flag is passed straight through to pipeline_cli.py (--help lists them).
set -euo pipefail

# =========================== EDIT THESE ==================================
# SOURCE: a single video file, a folder of videos, or a folder of frames.
SOURCE="/home/quidich/A_QuidichWork/matting-traindataprep/data/CCPL080626M1_1_16_1/camera04"

# OUT: where review/, crops/, masks/, final/ and preview/ are written.
OUT="/home/quidich/A_QuidichWork/matting-traindataprep/data/CCPL080626M1_1_16_1/camera04_out"

# BUDGET: "auto" (recommended) lets the footage decide how big the set is - it
# keeps taking the most-different person crop until the next one would be a
# near-repeat of what is already in the set. Static footage of one person ends
# up small, a busy varied match ends up large. Put a number here to force a
# fixed size instead.
BUDGET="auto"

# AUTO_FRAC tunes "auto": lower = fewer, more distinct crops; higher = more
# crops that overlap more. AUTO_MIN/AUTO_MAX are the floor and ceiling.
AUTO_FRAC="0.35"
AUTO_MIN="12"
AUTO_MAX="2000"

# PER_TRACK: max frames taken from one tracked person. Lower = more people,
# fewer shots each. MIN_GAP: frames that must separate two picks of one person.
PER_TRACK="6"
MIN_GAP="12"

# Detector gates. STRIDE reads every Nth video frame; MIN_HEIGHT drops people
# too small to matte; DHASH drops near-duplicate crops (bits, higher = stricter).
STRIDE="3"
MIN_CONF="0.3"
MIN_HEIGHT="96"
DHASH="6"

# yes = keep only whole bodies (head + knees + ankle, clear of the frame edge).
# no  = partial bodies are kept too.
FULL_BODY_ONLY="no"

# UNATTENDED=yes runs start to finish with no questions at all: every stage
# takes the settings above, and the review stage approves each mined frame
# instead of asking. Set it to "no" to get the prompts back.
UNATTENDED="yes"

# Anything else you want passed through, e.g. "--final-mode green --preview".
EXTRA=""
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
        PY="$SCRIPT_DIR/.venv/bin/python"
    elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
        PY="$SCRIPT_DIR/venv/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PY="$CONDA_PREFIX/bin/python"          # the activated conda env
    else
        PY="python3"
    fi
fi

echo "python : $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# --- dependency check -----------------------------------------------------
missing=""
check() {  # check <import name> <pip name>
    "$PY" - "$1" <<'PYCHK' >/dev/null 2>&1 || missing="$missing $2"
import importlib, sys
importlib.import_module(sys.argv[1])
PYCHK
}
check cv2 opencv-python
check numpy numpy
check PIL Pillow
check torch torch
check ultralytics ultralytics
check transformers transformers
check timm timm

if [ -n "$missing" ]; then
    echo "missing python package(s):$missing"
    echo "install them with:"
    echo "    $PY -m pip install -r requirements.txt"
    exit 1
fi
"$PY" -c "import clip" >/dev/null 2>&1 \
    || echo "note : CLIP not installed - the miner falls back to pose + scalars"
"$PY" -c "import torch; print('cuda   :', torch.cuda.is_available())"

# --- weights --------------------------------------------------------------
for w in weights/yolo11x.pt weights/yolo11x-pose.pt; do
    [ -f "$w" ] || echo "note : $w missing - ultralytics will download it"
done

# --- assemble the arguments ----------------------------------------------
# The variables above only fill in what the command line did not already give.
ARGS=()
case " $* " in
    *" --source "*) ;;
    *) if [ -n "$SOURCE" ]; then ARGS+=(--source "$SOURCE"); fi ;;
esac
case " $* " in
    *" --out "*) ;;
    *) if [ -n "$OUT" ]; then ARGS+=(--out "$OUT"); fi ;;
esac
# mining knobs
ARGS+=(--budget "$BUDGET" --auto-frac "$AUTO_FRAC"
       --auto-min "$AUTO_MIN" --auto-max "$AUTO_MAX"
       --per-track "$PER_TRACK" --min-gap "$MIN_GAP"
       --stride "$STRIDE" --min-conf "$MIN_CONF" --min-height "$MIN_HEIGHT"
       --dhash-dist "$DHASH")
case "$FULL_BODY_ONLY" in
    y|Y|yes|YES|true|1) ARGS+=(--full-body-only) ;;
esac
case "$UNATTENDED" in
    y|Y|yes|YES|true|1) ARGS+=(--yes --no-review) ;;
esac

if [ -n "$EXTRA" ]; then
    read -r -a _extra <<< "$EXTRA"
    ARGS+=("${_extra[@]}")
fi

if [ -n "$SOURCE" ] && [ ! -e "$SOURCE" ]; then
    case " $* " in
        *" --source "*) ;;
        *) echo "error : SOURCE does not exist: $SOURCE" >&2; exit 1 ;;
    esac
fi

printf 'source : %s\n' "$(case " $* " in *" --source "*) echo "<from command line>";; *) echo "${SOURCE:-<ask>}";; esac)"
printf 'output : %s\n' "$(case " $* " in *" --out "*) echo "<from command line>";; *) echo "${OUT:-<ask>}";; esac)"

exec "$PY" "$SCRIPT_DIR/pipeline_cli.py" "${ARGS[@]+"${ARGS[@]}"}" "$@"
