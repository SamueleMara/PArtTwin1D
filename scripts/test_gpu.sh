#!/usr/bin/env bash
# REAL integration tests on this workstation: ManiSkill/SAPIEN PhysX (CPU+GPU),
# gsplat CUDA rasterization, and the browser GUI (system Chrome via Playwright).
# Logs: outputs/logs/test_*.log ; evidence: outputs/test_artifacts/
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_sim.sh >/dev/null
mkdir -p outputs/logs
rc=0
PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/integration "$@" 2>&1 | tee outputs/logs/test_integration.log
rc=$(( rc | ${PIPESTATUS[0]} ))
PARTTWIN_RUN_GUI_TESTS=1 python -m pytest -v tests/gui "$@" 2>&1 | tee outputs/logs/test_gui.log
rc=$(( rc | ${PIPESTATUS[0]} ))
exit $rc
