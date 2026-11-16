#!/usr/bin/env bash
# Gaussians inside ManiSkill's SAPIEN viewer (desktop window).
# Usage: bash scripts/run_maniskill_gui.sh [config] [extra args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh >/dev/null
CONFIG="${1:-configs/examples/hinged_box.yaml}"
shift || true
exec python -m parttwin1d.cli.maniskill_gui --config "$CONFIG" "$@"
