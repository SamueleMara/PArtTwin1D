#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./setup_env.sh
python -m pytest -q tests
mkdir -p outputs
python -u -m parttwin_smoke.doctor --output outputs/gsplat_preflight.png
python -m parttwin_smoke.run \
  --manifest examples/synthetic_fixture/asset.json \
  --q-min 0 --q-max 1.3 --frames 20 --output outputs/synthetic_door
printf 'Output: %s\n' "$PWD/outputs/synthetic_door/sweep.gif"
