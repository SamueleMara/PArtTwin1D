#!/usr/bin/env bash
# Idempotent setup inside the EXISTING parttwin_sim environment (no new env,
# no torch/CUDA reinstall). Run from the repository root:
#   conda activate parttwin_sim && bash scripts/bootstrap_submodules.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh

echo "== submodules (pinned SHAs recorded in the superproject)"
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive

echo "== editable --no-deps installs (source runtime mode)"
python -m pip install --no-deps --no-build-isolation -e third_party/ManiSkill
python -m pip install --no-deps --no-build-isolation -e third_party/nerfview
# BUILD_NO_CUDA=1: do not compile at pip time; gsplat JIT-compiles once into
# ~/.cache/torch_extensions/py311_cu128/gsplat_cuda on first use.
BUILD_NO_CUDA=1 python -m pip install --no-deps --no-build-isolation -e third_party/gsplat
python -m pip install --no-deps --no-build-isolation -e .

echo "== small pure-Python runtime deps (pinned; see requirements/environment.lock.md)"
python -m pip install -r requirements/gui.in
python -m pip check

echo "== gsplat CUDA extension (first time: ~4 min JIT compile, then cached)"
python -m parttwin1d.cli.doctor --provenance --check-gsplat
