#!/usr/bin/env bash
# Launch the live 3D GUI.  Usage: bash scripts/run_gui.sh [config] [extra gui args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh >/dev/null
CONFIG="${1:-configs/examples/hinged_box.yaml}"
shift || true
exec python -m parttwin1d.cli.gui --config "$CONFIG" --host 127.0.0.1 "$@"
