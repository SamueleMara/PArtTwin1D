"""REAL ManiSkill 3.0.1 / SAPIEN 3.0.3 PhysX integration (no mocks, no robot).

Opt-in:  PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/integration
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import ROOT, base_config_dict, requires_gpu, write_config

pytestmark = [requires_gpu, pytest.mark.physx]


def run_checks(backend: str, cfg: Path) -> dict:
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("_physx_checks.py")), backend, str(cfg)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON ")]
    assert r.returncode == 0 and line, f"rc={r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return json.loads(line[-1][len("RESULT_JSON "):])


@pytest.fixture(scope="module", params=["physx_cpu", "physx_cuda"])
def results(request, tmp_path_factory):
    # Non-identity, explicitly configured root pose (applied before PhysX init).
    d = base_config_dict(ROOT)
    c, s = float(np.cos(0.6)), float(np.sin(0.6))
    d["T_world_root"] = [[c, -s, 0, 0.3], [s, c, 0, -0.2], [0, 0, 1, 0.15], [0, 0, 0, 1]]
    cfg = write_config(tmp_path_factory.mktemp(request.param) / "hinged_rooted.yaml", d)
    res = run_checks(request.param, cfg)
    out = ROOT / "outputs" / "test_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"physx_checks_{request.param}.json").write_text(json.dumps(res, indent=2))
    return res


def test_robot_free_env_api(results):
    assert results["agent_is_none"] and results["after_reconfigure_agent_none"]
    assert results["gpu_sim_enabled"] == (results["backend"] == "physx_cuda")
    assert results["obs_keys"] == ["agent", "extra"] and results["obs_agent_empty"]
    assert results["obs_extra_keys"] == ["qpos", "qvel"]
    assert "articulations" in results["state_keys"] and "controller" not in results["state_keys"]
    assert results["flat_state_dim"] > 0
    assert results["closed"]


def test_active_joint_discovery(results):
    assert results["joints"] == [["door_hinge", "revolute", "rad", 0.0, pytest.approx(1.57, abs=1e-6), "door_link"]]


def test_explicit_root_pose_applied(results):
    assert results["root_pose_error"] < 1e-5
    assert max(results["base_vs_T_world_root"]) < 1e-5
    assert results["root_pose_error_after_reconfigure"] < 1e-5


def test_kinematic_preview_matches_urdf_joint(results):
    assert results["kinematic_rot_err_max"] < 1e-4, results["kinematic_rot_err_max"]
    assert results["kinematic_trans_err_max"] < 1e-5
    assert results["pivot_drift_max"] < 1e-5           # hinge pivot stays fixed
    assert results["base_deviation_max"] < 1e-6        # fixed link unchanged
    assert max(results["return_to_q0_error"]) < 1e-5   # no accumulated drift
    assert results["kinematic_sim_time"] == 0.0        # kinematic preview is not a trajectory
    assert results["limit_violation_rejected"]


def test_state_dict_roundtrip(results):
    assert results["state_roundtrip_q"] == pytest.approx(0.9, abs=1e-5)


def test_gpu_resident_link_pose_tensor(results):
    lp = results["link_poses_tensor"]
    assert lp["shape"] == [1, 2, 4, 4]
    assert lp["device"].startswith("cuda" if results["backend"] == "physx_cuda" else "cpu")
    assert results["link_poses_tensor_matches_snapshot"] < 1e-5


def test_dynamics_preview_is_separate_and_physical(results):
    assert results["pd_sim_time"] == pytest.approx(2.0, rel=1e-6)   # 100 control steps @ 50 Hz
    assert 1.0 < results["pd_final_q"] <= 1.2 + 1e-3                 # PD drive converges toward 1.2
    assert results["pd_monotone_frac"] > 0.95
    assert results["base_deviation_after_dynamics"] < 1e-5           # fixed base stays fixed
    assert results["zero_force_drift"] < 1e-3                        # vertical hinge: no gravity torque
    assert results["positive_torque_q"] > 0.5 + 1e-3                 # +torque opens the door
