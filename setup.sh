#!/usr/bin/env bash
# One script to make this folder runnable from scratch.
#
#   ./setup.sh                 # packages + every weight the pipeline needs
#   ./setup.sh --venv          # build a project-local .venv and install into it
#   ./setup.sh --no-venv       # install into whatever python is active
#   ./setup.sh --no-weights    # packages only
#   ./setup.sh --sam3          # also try SAM 3 (gated, needs `hf auth login`)
#   ./setup.sh --check         # change nothing, just report what is missing
#
# Which python gets used, in order: $PYTHON if you set it, then an existing
# ./.venv, then --venv/--no-venv, then an active conda env, and failing all of
# that a fresh ./.venv is created for you. The run_*.sh scripts prefer ./.venv
# too, so once it exists everything lines up on its own.
#
# It is safe to re-run: anything already present is left alone.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WANT_WEIGHTS=1
WANT_SAM3=0
CHECK_ONLY=0
VENV_MODE=auto          # auto | force | never
VENV_DIR="$SCRIPT_DIR/.venv"
for arg in "$@"; do
    case "$arg" in
        --no-weights) WANT_WEIGHTS=0 ;;
        --sam3)       WANT_SAM3=1 ;;
        --check)      CHECK_ONLY=1; WANT_WEIGHTS=0 ;;
        --venv)       VENV_MODE=force ;;
        --no-venv)    VENV_MODE=never ;;
        -h|--help)    sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
    esac
done

ok=0; warn=0; fail=0
say()  { printf '  %s\n' "$*"; }
good() { printf '  \033[32m✓\033[0m %s\n' "$*"; ok=$((ok+1)); }
note() { printf '  \033[33m!\033[0m %s\n' "$*"; warn=$((warn+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=$((fail+1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- python ---
head_ "1/5  python"
make_venv() {  # make_venv <base python>
    say "creating a project venv in .venv (python -m venv) …"
    if ! "$1" -m venv "$VENV_DIR" 2>/tmp/venv_err.$$; then
        bad "could not create .venv:"
        sed 's/^/      /' /tmp/venv_err.$$ >&2
        if grep -qi "ensurepip\|python3-venv" /tmp/venv_err.$$; then
            say "on Debian/Ubuntu: sudo apt install python3-venv"
        fi
        rm -f /tmp/venv_err.$$
        rm -rf "$VENV_DIR"
        return 1
    fi
    rm -f /tmp/venv_err.$$
    "$VENV_DIR/bin/python" -m pip install -q --upgrade pip setuptools wheel \
        || note "could not upgrade pip inside .venv"
    return 0
}

PY="${PYTHON:-}"
CREATED_VENV=0
if [ -n "$PY" ]; then
    say "using \$PYTHON"
elif [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"                       # the project venv already exists
elif [ "$VENV_MODE" = never ]; then
    PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
    [ -x "${PY:-}" ] || PY="$(command -v python3 || true)"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    # --check must not build anything; report against the active interpreter
    PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
    [ -x "${PY:-}" ] || PY="$(command -v python3 || true)"
    note "no .venv yet — checking the active python instead. ./setup.sh --venv builds one"
else
    BASE_PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
    [ -x "${BASE_PY:-}" ] || BASE_PY="$(command -v python3 || true)"
    if [ -z "$BASE_PY" ]; then
        bad "no python3 on PATH to build a venv from"
        exit 1
    fi
    if [ "$VENV_MODE" = auto ] && [ -n "${CONDA_PREFIX:-}" ]; then
        # An active conda env probably already carries torch and friends;
        # silently rebuilding that in a venv would re-download gigabytes.
        PY="$BASE_PY"
        note "conda env active ($(basename "$CONDA_PREFIX")) — installing there.
       ./setup.sh --venv builds a project-local .venv instead"
    else
        make_venv "$BASE_PY" || exit 1
        PY="$VENV_DIR/bin/python"
        CREATED_VENV=1
        good "created .venv (the run_*.sh scripts pick it up automatically)"
    fi
fi

if [ -z "$PY" ] || ! "$PY" -c '' 2>/dev/null; then
    bad "no usable python found. Activate an env, or run ./setup.sh --venv"
    exit 1
fi
good "$("$PY" -c 'import sys; print(sys.executable)')  ($("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))'))"
if [ "$CREATED_VENV" -eq 1 ]; then
    say "torch and its CUDA wheels are a few GB — the first install is slow"
fi

# -------------------------------------------------------------- packages ---
head_ "2/5  python packages"
if [ "$CHECK_ONLY" -eq 0 ]; then
    say "pip install -r requirements.txt (quiet; only missing packages are fetched)"
    if "$PY" -m pip install -q -r requirements.txt; then
        good "requirements.txt satisfied"
    else
        bad "pip install failed — scroll up for the reason"
    fi
    # Optional, and the miner degrades gracefully without it.
    if ! "$PY" -c 'import clip' >/dev/null 2>&1; then
        say "installing CLIP (appearance diversity for the miner)"
        "$PY" -m pip install -q git+https://github.com/openai/CLIP.git \
            && good "CLIP installed" \
            || note "CLIP failed to install — the miner falls back to pose + scalars"
    fi
fi

"$PY" - <<'PYCHK'
import importlib.util, sys
need = [("cv2","opencv-python"), ("numpy","numpy"), ("PIL","Pillow"),
        ("torch","torch"), ("ultralytics","ultralytics"), ("lap","lap"),
        ("transformers","transformers"), ("timm","timm"), ("einops","einops"),
        ("kornia","kornia"), ("safetensors","safetensors"),
        ("PySide6","PySide6"), ("yaml","PyYAML"), ("huggingface_hub","huggingface_hub")]
missing = [pipname for mod, pipname in need
           if not importlib.util.find_spec(mod)]
if missing:
    print("  \033[31m✗\033[0m missing: " + " ".join(missing))
    sys.exit(1)
print("  \033[32m✓\033[0m every required package imports")

import transformers
sam = [n for n in ("Sam2Model", "Sam2Processor") if not hasattr(transformers, n)]
if sam:
    print(f"  \033[31m✗\033[0m transformers {transformers.__version__} lacks {sam} "
          f"- pip install -U 'transformers>=5.12'")
    sys.exit(1)
print(f"  \033[32m✓\033[0m transformers {transformers.__version__} ships SAM 2"
      + (" and SAM 3" if hasattr(transformers, "Sam3Model") else ""))
if importlib.util.find_spec("clip"):
    print("  \033[32m✓\033[0m CLIP present (appearance diversity on)")
else:
    print("  \033[33m!\033[0m CLIP absent - the miner uses pose + scalars only")
PYCHK
[ $? -eq 0 ] && ok=$((ok+1)) || fail=$((fail+1))

# ------------------------------------------------------------------ cuda ---
head_ "3/5  gpu"
"$PY" - <<'PYCUDA'
try:
    import torch
except ImportError:
    raise SystemExit("  \033[33m!\033[0m torch is not installed yet - "
                     "cannot report the gpu")
if torch.cuda.is_available():
    print(f"  \033[32m✓\033[0m cuda {torch.version.cuda}: "
          f"{torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
else:
    print("  \033[33m!\033[0m no cuda - everything still runs, just slowly")
PYCUDA

# --------------------------------------------------------------- weights ---
head_ "4/5  weights"
mkdir -p weights

get_yolo() {  # get_yolo <file>
    if [ -f "weights/$1" ]; then
        good "weights/$1 ($(du -h "weights/$1" | cut -f1))"
        return
    fi
    if [ "$WANT_WEIGHTS" -eq 0 ]; then note "weights/$1 missing"; return; fi
    say "downloading $1 …"
    if "$PY" -c "
from ultralytics import YOLO
import shutil, pathlib
m = YOLO('$1')
p = pathlib.Path(getattr(m, 'ckpt_path', '') or '$1')
if p.exists() and p.resolve() != pathlib.Path('weights/$1').resolve():
    shutil.move(str(p), 'weights/$1')
" >/dev/null 2>&1 && [ -f "weights/$1" ]; then
        good "weights/$1"
    else
        note "could not fetch $1 — ultralytics will download it on first run"
    fi
}

get_yolo yolo11x.pt
get_yolo yolo11x-pose.pt

# BiRefNet: not gated, but it is a transformers repo rather than a single file.
if "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('ZhengPeng7/BiRefNet', allow_patterns=['*.json','*.py','*.safetensors'])
" >/dev/null 2>&1; then
    good "BiRefNet cached (matting model for the masks stage)"
else
    if [ "$WANT_WEIGHTS" -eq 1 ]; then
        note "BiRefNet not cached — the masks stage will download it on first run"
    fi
fi

# SAM 2.1: ungated, used by the mask fixer UI.
SAM2_DIR="weights/sam2.1-hiera-large"
if [ -f "$SAM2_DIR/model.safetensors" ]; then
    good "$SAM2_DIR ($(du -sh "$SAM2_DIR" | cut -f1))"
elif [ "$WANT_WEIGHTS" -eq 1 ]; then
    say "downloading SAM 2.1 hiera-large (~860 MB) …"
    if "$PY" - <<'PYSAM2' ; then
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam2.1-hiera-large",
                  local_dir="weights/sam2.1-hiera-large",
                  allow_patterns=["*.json", "*.safetensors"])
PYSAM2
        good "$SAM2_DIR"
    else
        bad "SAM 2.1 download failed — the mask fixer UI will not run"
    fi
else
    note "$SAM2_DIR missing (mask fixer needs it)"
fi

# SAM 3: gated. Only attempted on request, and never fatal.
if [ "$WANT_SAM3" -eq 1 ]; then
    if [ -f "weights/sam3/model.safetensors" ]; then
        good "weights/sam3"
    else
        say "trying SAM 3 (gated) …"
        # stderr is dropped: being gated is the expected outcome here and the
        # note below says everything the traceback would.
        if "$PY" - 2>/dev/null <<'PYSAM3' ; then
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam3", local_dir="weights/sam3",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
PYSAM3
            good "weights/sam3 — you can set model: sam3 in mask_fixer.yaml"
        else
            note "SAM 3 is gated and not authorised on this machine:
       1. accept the licence at https://huggingface.co/facebook/sam3
       2. hf auth login   (or export HF_TOKEN=hf_...)
       3. re-run ./setup.sh --sam3
       Meanwhile the mask fixer works on SAM 2."
        fi
    fi
fi

# --------------------------------------------------------------- wiring ----
head_ "5/5  scripts and config"
for f in run_pipeline.sh run_mask_fixer.sh run_extract_frames.sh setup_sam.sh; do
    if [ -f "$f" ]; then
        [ -x "$f" ] || chmod +x "$f"
        good "$f"
    else
        note "$f is missing"
    fi
done

"$PY" - <<'PYCFG'
from pathlib import Path
import sys
sys.path.insert(0, ".")
try:
    from mask_fixer import load_config
    cfg = load_config(Path("mask_fixer.yaml"))
except Exception as e:
    print(f"  \033[31m✗\033[0m mask_fixer.yaml does not load: {e}")
    sys.exit(0)
print(f"  \033[32m✓\033[0m mask_fixer.yaml loads (model: {cfg.get('model')})")
for key in ("images", "masks"):
    p = Path(cfg[key])
    mark, col = ("✓", "32") if p.is_dir() else ("!", "33")
    tail = "" if p.is_dir() else "  (make it with ./run_pipeline.sh)"
    print(f"  \033[{col}m{mark}\033[0m {key}: {p}{tail}")
PYCFG

grep -q '^SOURCE="' run_pipeline.sh 2>/dev/null && \
    say "run_pipeline.sh SOURCE = $(grep -m1 '^SOURCE=' run_pipeline.sh | cut -d'"' -f2)"

# ----------------------------------------------------------------- done ----
printf '\n\033[1msummary\033[0m  %d ok, %d warning(s), %d problem(s)\n' "$ok" "$warn" "$fail"
if [ "$fail" -eq 0 ]; then
    if [ -x "$VENV_DIR/bin/python" ]; then
        printf '\nthe project venv is at .venv — run_pipeline.sh and run_mask_fixer.sh\n'
        printf 'use it on their own. For your own commands:  source .venv/bin/activate\n'
    fi
    cat <<'EOF'

ready. the usual order:

  1. edit SOURCE and OUT at the top of run_pipeline.sh
  2. ./run_pipeline.sh              mine -> crops -> BiRefNet mattes -> cutouts
  3. edit images/masks in mask_fixer.yaml to match that OUT folder
  4. ./run_mask_fixer.sh            fix what the matte dropped, with SAM 2
EOF
else
    echo
    echo "fix the ✗ items above, then re-run ./setup.sh"
    exit 1
fi
