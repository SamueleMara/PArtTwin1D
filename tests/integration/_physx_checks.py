"""Real ManiSkill 3.0.1 / SAPIEN 3.0.3 checks, run in a fresh process per backend.

    python tests/integration/_physx_checks.py physx_cpu|physx_cuda <config.yaml>
Prints one JSON object with measured values; assertions live in the pytest file.
(GPU PhysX state is process-global in SAPIEN, hence one process per backend.)
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(backend: str, cfg_path: str) -> dict:
    import torch

    from parttwin1d.config import load_config
    from parttwin1d.evaluation.consistency import expected_child_pose, pose_error
    from parttwin1d.sim.articulation import ArticulationDriver
    from parttwin1d.sim.env import create_env, root_pose_error
    from parttwin1d.sim.state import link_poses_tensor

    cfg = load_config(cfg_path)
    out = {"backend": backend}
    env = create_env(cfg, backend)
    obs, info = env.reset(seed=0)
    out["agent_is_none"] = env.agent is None
    out["gpu_sim_enabled"] = bool(env.gpu_sim_enabled)
    out["obs_keys"] = sorted(obs.keys())
    out["obs_extra_keys"] = sorted(obs["extra"].keys())
    out["obs_agent_empty"] = obs["agent"] == {}
    sd = env.get_state_dict()
    out["state_keys"] = sorted(sd.keys())
    out["flat_state_dim"] = int(env.get_state().shape[-1])
    out["root_pose_error"] = root_pose_error(env, cfg)
    d = ArticulationDriver(env, cfg)
    out["joints"] = [(j.name, j.type, j.unit, j.lower, j.upper, j.child_link) for j in d.joints]
    out["all_urdf_joints"] = sorted(d.urdf.joints)
    s0 = d.snapshot("reset")
    uj = d.urdf.joints["door_hinge"]
    T0 = s0.T_world_link["door_link"]
    base0 = s0.T_world_link["base_link"]
    out["base_vs_T_world_root"] = pose_error(base0, cfg.T_world_root)
    rot_err, tr_err, base_dev, pivot = [], [], [], []
    for q in [0.3, 1.0, 1.57, 0.05, 0.7, 0.0]:
        d.set_joint_kinematic("door_hinge", q)
        s = d.snapshot("kinematic")
        e = pose_error(expected_child_pose(T0, uj, q, 0.0), s.T_world_link["door_link"])
        tr_err.append(e[0])
        rot_err.append(e[1])
        base_dev.append(max(pose_error(base0, s.T_world_link["base_link"])))
        pivot.append(float(np.linalg.norm(s.T_world_link["door_link"][:3, 3] - T0[:3, 3])))
        assert abs(s.q[0] - q) < 1e-5
    out["kinematic_rot_err_max"] = max(rot_err)
    out["kinematic_trans_err_max"] = max(tr_err)
    out["base_deviation_max"] = max(base_dev)
    out["pivot_drift_max"] = max(pivot)
    out["kinematic_sim_time"] = d.sim_time
    out["return_to_q0_error"] = pose_error(T0, d.snapshot("kinematic").T_world_link["door_link"])
    try:
        d.set_joint_kinematic("door_hinge", 2.0)
        out["limit_violation_rejected"] = False
    except ValueError:
        out["limit_violation_rejected"] = True

    # state dict round trip (robot-free)
    d.set_joint_kinematic("door_hinge", 0.9)
    st = env.get_state_dict()
    d.set_joint_kinematic("door_hinge", 0.2)
    env.set_state_dict(st)
    out["state_roundtrip_q"] = float(d.get_q()[0])

    # batched GPU-resident link poses (future PPO path)
    lp = link_poses_tensor(env, cfg.link_names)
    out["link_poses_tensor"] = {"shape": list(lp.shape), "device": str(lp.device)}
    out["link_poses_tensor_matches_snapshot"] = float(
        np.abs(lp[0, 1].detach().cpu().numpy() - d.snapshot("kinematic").T_world_link["door_link"]).max())

    # dynamics: PD drive and generalized force, separate from kinematic preview
    d.set_joint_kinematic("door_hinge", 0.0)
    t_before = d.sim_time
    traj = []
    for _ in range(100):
        d.step_dynamics(1, drive_target=[1.2])
        traj.append(float(d.get_q()[0]))
    out["pd_final_q"] = traj[-1]
    out["pd_monotone_frac"] = float(np.mean(np.diff(traj) >= -1e-4))
    out["pd_sim_time"] = d.sim_time - t_before
    out["base_deviation_after_dynamics"] = max(pose_error(base0, d.snapshot("dynamics").T_world_link["base_link"]))
    d.set_joint_kinematic("door_hinge", 0.5)
    for _ in range(25):
        d.step_dynamics(1, joint_force=[0.0])
    out["zero_force_drift"] = float(abs(d.get_q()[0] - 0.5))
    for _ in range(25):
        d.step_dynamics(1, joint_force=[1.0])
    out["positive_torque_q"] = float(d.get_q()[0])

    # reconfigure / reset keeps the robot-free scene valid
    env.reset(seed=1, options={"reconfigure": True})
    d2 = ArticulationDriver(env, cfg)
    out["after_reconfigure_q"] = d2.get_q().tolist()
    out["after_reconfigure_agent_none"] = env.agent is None
    out["root_pose_error_after_reconfigure"] = root_pose_error(env, cfg)
    env.close()
    out["closed"] = True
    out["torch_cuda"] = torch.cuda.is_available()
    return out


if __name__ == "__main__":
    print("RESULT_JSON " + json.dumps(main(sys.argv[1], sys.argv[2])))
