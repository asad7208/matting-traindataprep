#!/usr/bin/env bash
# SAM 3 mask repair UI.
#
#   ./run_mask_fixer.sh                 # uses mask_fixer.yaml
#   ./run_mask_fixer.sh other.yaml      # uses another config
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-$SCRIPT_DIR/mask_fixer.yaml}"

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

missing=""
for pkg in PySide6 numpy PIL transformers yaml; do
    "$PY" -c "import $pkg" >/dev/null 2>&1 || missing="$missing $pkg"
done
if [ -n "$missing" ]; then
    echo "missing:$missing"
    echo "run ./setup_sam.sh first"
    exit 1
fi

[ -f "$CONFIG" ] || { echo "no config file: $CONFIG" >&2; exit 1; }
echo "config : $CONFIG"
exec "$PY" "$SCRIPT_DIR/mask_fixer.py" "$CONFIG"
