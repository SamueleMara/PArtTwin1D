#!/usr/bin/env bash
# Provenance + compatibility report (exit 1 on any incompatibility).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh >/dev/null
mkdir -p outputs/logs
python -m parttwin1d.cli.doctor --provenance --check-gsplat --json outputs/logs/doctor.json "$@"
