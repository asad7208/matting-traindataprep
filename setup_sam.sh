#!/usr/bin/env bash
# One-time SAM setup: python packages + weights into ./weights
#
#   ./setup_sam.sh          # SAM 2.1, no licence needed  (the default)
#   ./setup_sam.sh sam3     # SAM 3, needs the gated licence first:
#                           #   1. accept at https://huggingface.co/facebook/sam3
#                           #   2. hf auth login   (or export HF_TOKEN=hf_...)
set -euo pipefail

WHICH="${1:-sam2}"
case "$WHICH" in
    sam2) REPO="${SAM_REPO:-facebook/sam2.1-hiera-large}" ;;
    sam3) REPO="${SAM_REPO:-facebook/sam3}" ;;
    *)    echo "usage: $0 [sam2|sam3]" >&2; exit 1 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/weights/$(basename "$REPO")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PY="$CONDA_PREFIX/bin/python"
    else
        PY="python3"
    fi
fi
echo "python : $("$PY" -c 'import sys; print(sys.executable)')"

# --- packages -------------------------------------------------------------
# SAM 3 ships inside transformers itself; nothing else to build.
echo "checking packages …"
"$PY" -m pip install -q "transformers>=5.12" "huggingface_hub>=1.5" \
                        "PySide6>=6.5" "pyyaml" "safetensors>=0.4" "timm>=0.9"

"$PY" - "$WHICH" <<'PYCHK'
import sys, transformers
need = {"sam2": ("Sam2Model", "Sam2Processor"),
        "sam3": ("Sam3Model", "Sam3Processor",
                 "Sam3TrackerModel", "Sam3TrackerProcessor")}[sys.argv[1]]
missing = [n for n in need if not hasattr(transformers, n)]
if missing:
    raise SystemExit(f"transformers {transformers.__version__} has no {missing} "
                     f"- upgrade with: pip install -U transformers")
print(f"transformers {transformers.__version__}: {', '.join(need)} present")
PYCHK

# --- weights --------------------------------------------------------------
echo "downloading $REPO -> $DEST"
"$PY" - "$REPO" "$DEST" <<'PYGET'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
try:
    path = snapshot_download(
        repo_id=repo, local_dir=dest,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
        raise SystemExit(
            f"\n{repo} is gated and this machine is not authorised yet.\n"
            f"  1. open https://huggingface.co/{repo} and accept the licence\n"
            f"  2. hf auth login   (or export HF_TOKEN=hf_...)\n"
            f"  3. re-run ./setup_sam.sh sam3\n"
            f"\nSAM 2 needs no licence: ./setup_sam.sh\n")
    raise SystemExit(f"download failed: {msg}")
print(f"\nweights in {path}")
PYGET

echo
echo "done. mask_fixer.yaml already points at weights/$(basename "$REPO")"
echo "then run:  ./run_mask_fixer.sh"
