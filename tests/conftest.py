"""Shared fixtures. Real integration tests are opt-in:

    PARTTWIN_RUN_GPU_TESTS=1  -> tests/integration (real ManiSkill/SAPIEN PhysX + gsplat CUDA)
    PARTTWIN_RUN_GUI_TESTS=1  -> tests/gui (real browser via Playwright + system Chrome)

Skipped tests are reported as SKIPPED, never as passes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HINGED = ROOT / "configs" / "examples" / "hinged_box.yaml"

RUN_GPU = os.environ.get("PARTTWIN_RUN_GPU_TESTS") == "1"
RUN_GUI = os.environ.get("PARTTWIN_RUN_GUI_TESTS") == "1"

requires_gpu = pytest.mark.skipif(not RUN_GPU, reason="opt-in: set PARTTWIN_RUN_GPU_TESTS=1 (real PhysX + gsplat CUDA)")
requires_gui = pytest.mark.skipif(not RUN_GUI, reason="opt-in: set PARTTWIN_RUN_GUI_TESTS=1 (real browser)")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def hinged_cfg_path() -> Path:
    return HINGED


@pytest.fixture(scope="session")
def hinged_cfg():
    from parttwin1d.config import load_config

    return load_config(HINGED)


def write_config(path: Path, data: dict) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def base_config_dict(root: Path) -> dict:
    import yaml

    d = yaml.safe_load(HINGED.read_text())
    ex = root / "examples" / "hinged_box"
    d["urdf"] = str(ex / "model.urdf")
    d["links"]["base_link"]["ply"] = str(ex / "base.ply")
    d["links"]["door_link"]["ply"] = str(ex / "door.ply")
    return d
