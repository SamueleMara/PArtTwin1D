"""Joint discovery and the two explicitly separated command modes.

``kinematic``  set q directly through ManiSkill/SAPIEN (``set_qpos``), zero
               qvel, and on GPU apply + update articulation kinematics + fetch.
               No time passes; this is NOT a simulated trajectory.
``dynamics``   advance PhysX with ``env.step(None)`` while an explicit,
               labelled external input acts on the joints: either a PD joint
               drive target or a generalized joint force (``qf``). Gravity,
               URDF inertia/damping/friction and contacts are simulated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assets.urdf import parse_urdf
from .state import StateSnapshot, capture_snapshot


@dataclass(frozen=True)
class JointInfo:
    index: int
    name: str
    type: str
    unit: str
    lower: float
    upper: float
    child_link: str
    axis_in_child: np.ndarray  # unit axis in the joint/child-link frame (URDF)

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.lower) and np.isfinite(self.upper))


class ArticulationDriver:
    """Owns command application to one robot-free ``ArticulatedTwinEnv``.

    Not thread-safe by design: exactly one thread (the simulation owner, see
    ``viewer.coordination``) may call it.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.art = env.twin
        self.urdf = parse_urdf(cfg.urdf)
        self.bound_links = cfg.link_names
        absent = [link for link in self.bound_links if link not in self.art.links_map]
        if absent:
            raise ValueError(f"Gaussian-bound links {absent} missing in PhysX articulation {sorted(self.art.links_map)}")
        self._active = list(self.art.get_active_joints())
        lim = self.art.get_qlimits()[0].detach().cpu().numpy()
        self.joints: list[JointInfo] = []
        for i, j in enumerate(self._active):
            uj = self.urdf.joints.get(j.name)
            if uj is None:
                raise ValueError(f"PhysX active joint {j.name!r} not found in URDF")
            self.joints.append(JointInfo(i, j.name, uj.type, uj.unit, float(lim[i, 0]), float(lim[i, 1]),
                                         uj.child, uj.axis.copy()))
        self.joint_index = {j.name: j.index for j in self.joints}
        self.sim_time = 0.0
        self.revision = 0
        self.q0 = self.get_q()

    def close(self) -> None:
        """Close the ManiSkill scene (call from the owning thread)."""
        if self.env is not None:
            self.env.close()
            self.env = None

    # ------------------------------------------------------------------ queries
    @property
    def gpu(self) -> bool:
        return bool(self.env.gpu_sim_enabled)

    @property
    def dof(self) -> int:
        return len(self.joints)

    def get_q(self) -> np.ndarray:
        return self.art.get_qpos()[0].detach().cpu().numpy().astype(np.float64)

    def joint_meta(self) -> dict:
        return {j.name: (j.type, j.unit) for j in self.joints}

    def snapshot(self, mode: str, note: str = "", request_id: int = -1) -> StateSnapshot:
        self.revision += 1
        return capture_snapshot(self.env, self.bound_links, revision=self.revision, sim_time=self.sim_time,
                                mode=mode, joint_meta=self.joint_meta(), note=note, request_id=request_id)

    # --------------------------------------------------------------- kinematics
    def _check_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, np.float64).reshape(self.dof)
        if not np.isfinite(q).all():
            raise ValueError(f"non-finite joint values {q}")
        for j in self.joints:
            if j.bounded and not (j.lower - 1e-6 <= q[j.index] <= j.upper + 1e-6):
                raise ValueError(f"joint {j.name}={q[j.index]:.6f} {j.unit} outside URDF limits "
                                 f"[{j.lower:.6f}, {j.upper:.6f}]")
        return q

    def set_q_kinematic(self, q) -> None:
        """Teleport the articulation to q (no time advance)."""
        import torch

        q = self._check_q(q)
        qt = torch.as_tensor(q[None], dtype=torch.float32, device=self.env.device)
        self.art.set_qpos(qt)
        self.art.set_qvel(torch.zeros_like(qt))
        self._set_drive_targets(qt)
        if self.gpu:
            # Same sequence as BaseEnv.set_state_dict in ManiSkill 3.0.1.
            sc = self.env.scene
            sc._gpu_apply_all()
            sc.px.gpu_update_articulation_kinematics()
            sc._gpu_fetch_all()
        # CPU: SAPIEN updates link poses synchronously inside set_qpos
        # (verified by tests/integration/test_maniskill_robotless.py).

    def set_joint_kinematic(self, name: str, value: float) -> None:
        q = self.get_q()
        if name not in self.joint_index:
            raise KeyError(f"unknown active joint {name!r}; active: {list(self.joint_index)}")
        q[self.joint_index[name]] = value
        self.set_q_kinematic(q)

    def reset_to_q0(self) -> None:
        self.set_q_kinematic(self.q0)
        self.sim_time = 0.0

    # ----------------------------------------------------------------- dynamics
    def _set_drive_targets(self, qt) -> None:
        self.art.set_joint_drive_targets(qt, joints=self._active)
        if self.gpu:
            self.env.scene.px.gpu_apply_articulation_target_position()

    def step_dynamics(self, n_control_steps: int = 1, *, drive_target=None, joint_force=None) -> None:
        """Advance PhysX by n control steps under an explicit external input.

        drive_target: q* for the PD joint drives (stiffness/damping from config).
        joint_force:  generalized joint force/torque [D] (N or N*m); when given,
                      drive targets track the current q (pure damping).
        """
        import torch

        if drive_target is not None and joint_force is not None:
            raise ValueError("choose either drive_target or joint_force")
        dev = self.env.device
        for _ in range(int(n_control_steps)):
            if drive_target is not None:
                self._set_drive_targets(torch.as_tensor(self._check_q(drive_target)[None], dtype=torch.float32, device=dev))
                qf = torch.zeros((1, self.dof), dtype=torch.float32, device=dev)
            else:
                self._set_drive_targets(self.art.get_qpos().clone())
                f = np.zeros(self.dof) if joint_force is None else np.asarray(joint_force, np.float64).reshape(self.dof)
                qf = torch.as_tensor(f[None], dtype=torch.float32, device=dev)
            self.art.set_qf(qf)
            if self.gpu:
                self.env.scene.px.gpu_apply_articulation_qf()
            self.env.step(None)
            self.sim_time += self.env.control_timestep
