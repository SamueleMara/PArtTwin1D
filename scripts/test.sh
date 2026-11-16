#!/usr/bin/env bash
# CPU unit tests + collection of every production test (GPU/GUI tests SKIP unless opted in).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh >/dev/null   # unsets PYTHONPATH (ROS), sets PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q "$@"
