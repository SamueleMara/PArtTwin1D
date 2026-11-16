"""Immutable articulation snapshots consumed by the renderer and the GUI.

Two paths:

* :class:`StateSnapshot` — single scene, NumPy float64, read-only arrays. This
  is the UI/export path and crosses the GPU->CPU boundary once per published
  state (a handful of 4x4 matrices), never per Gaussian.
* :func:`link_poses_tensor` — batched ``[B, L, 4, 4]`` torch tensor on the sim
  device with **no** host transfer; this is the interface intended for a future
  vectorized visual-PPO observation path (not benchmarked as such yet).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry.se3 import world_from_maniskill_pose


def _ro(a) -> np.ndarray:
    a = np.array(a, dtype=np.float64, copy=True)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class JointStatus:
    name: str
    type: str          # revolute | prismatic | continuous
    unit: str          # rad | m
    lower: float
    upper: float
    q: float
    dq: float
    at_lower: bool
    at_upper: bool
    bounded: bool


@dataclass(frozen=True)
class StateSnapshot:
    revision: int
    sim_time: float               # simulated seconds since reset (dynamics steps only)
    wall_time: float              # time.time() at capture
    env_index: int
    mode: str                     # "kinematic" | "dynamics" | "reset"
    joint_names: tuple[str, ...]
    q: np.ndarray                 # [D] read-only
    dq: np.ndarray                # [D] read-only
    qlimits: np.ndarray           # [D, 2] read-only
    T_world_link: dict = field(default_factory=dict)  # link -> read-only 4x4
    joints: tuple[JointStatus, ...] = ()
    note: str = ""
    request_id: int = -1          # last UI/command request applied before capture

    def q_dict(self) -> dict:
        return {n: float(v) for n, v in zip(self.joint_names, self.q)}

    def to_json(self) -> dict:
        return {
            "revision": self.revision, "sim_time": self.sim_time, "wall_time": self.wall_time,
            "env_index": self.env_index, "mode": self.mode, "q": self.q_dict(),
            "dq": {n: float(v) for n, v in zip(self.joint_names, self.dq)},
            "qlimits": {n: [float(a), float(b)] for n, (a, b) in zip(self.joint_names, self.qlimits)},
            "T_world_link": {k: v.tolist() for k, v in self.T_world_link.items()},
            "note": self.note, "request_id": self.request_id,
        }


def capture_snapshot(env, links, *, revision: int, sim_time: float, mode: str,
                     joint_meta: dict | None = None, env_index: int = 0, note: str = "",
                     request_id: int = -1) -> StateSnapshot:
    """Read q, dq and T_world_link for every bound link from the live articulation."""
    art = env.twin
    names = tuple(j.name for j in art.get_active_joints())
    q = art.get_qpos()[env_index].detach().cpu().numpy()
    dq = art.get_qvel()[env_index].detach().cpu().numpy()
    lim = art.get_qlimits()[env_index].detach().cpu().numpy()
    T = {}
    for link in links:
        if link not in art.links_map:
            raise KeyError(f"bound link {link!r} not in articulation links {sorted(art.links_map)}")
        T[link] = _ro(world_from_maniskill_pose(art.links_map[link].pose, env_index))
    joints = []
    meta = joint_meta or {}
    for i, n in enumerate(names):
        lo, hi = float(lim[i, 0]), float(lim[i, 1])
        bounded = bool(np.isfinite(lo) and np.isfinite(hi))
        jtype, unit = meta.get(n, ("revolute", "rad"))
        tol = 1e-4
        joints.append(JointStatus(n, jtype, unit, lo, hi, float(q[i]), float(dq[i]),
                                  bounded and q[i] <= lo + tol, bounded and q[i] >= hi - tol, bounded))
    return StateSnapshot(revision, float(sim_time), time.time(), env_index, mode, names,
                         _ro(q), _ro(dq), _ro(lim), T, tuple(joints), note, request_id)


def link_poses_tensor(env, links):
    """Batched T_world_link [B, L, 4, 4] on the simulation device (no host copy)."""
    import torch

    from ..geometry.se3 import pose_tensor_to_matrix

    art = env.twin
    mats = []
    for link in links:
        pose = art.links_map[link].pose
        mats.append(pose_tensor_to_matrix(pose.p, pose.q))
    return torch.stack(mats, dim=1)
