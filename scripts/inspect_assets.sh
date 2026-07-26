#!/usr/bin/env bash
# Run scripts/inspect_assets.py under the pinned packman USD runtime.
#
# `pxr` is importable from NEITHER default interpreter (kit python lacks it
# outside a running app; system python3 lacks it entirely), so offline USD
# inspection needs this exact invocation. Wrapping it here keeps the incantation
# in one place instead of pasted into three docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

USDROOT="$(ls -d "$HOME"/.cache/packman/chk/usd.py311.manylinux_2_35_aarch64.stock.release/* 2>/dev/null | head -1)"
if [[ -z "$USDROOT" ]]; then
  echo "FATAL: packman USD runtime not found. Has Isaac Sim been run at least once?" >&2
  exit 1
fi

PYTHONPATH="$USDROOT/lib/python" \
LD_LIBRARY_PATH="$USDROOT/lib:${LD_LIBRARY_PATH:-}" \
  exec "$HOME/IsaacLab/_isaac_sim/python.sh" "$REPO_ROOT/scripts/inspect_assets.py" "$@"
