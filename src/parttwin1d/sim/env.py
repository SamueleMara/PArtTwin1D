"""Robot-free ManiSkill 3.0.1 scene holding exactly one articulated URDF object.

* Loaded through ManiSkill's own URDF loader (SAPIEN/PhysX articulation).
* No agent: ``robot_uids="none"``; no dummy robot is instantiated.
* ``render_backend="none"``: all appearance comes from gsplat. PhysX collision
  and inertial records stay intact; only URDF *visual* records are dropped.
* The fixed root pose ``T_world_root`` is applied to the articulation builder
  *before* PhysX/GPU initialization.

ManiSkill 3.0.1 robot-free compatibility (audited in
``third_party/ManiSkill/mani_skill/envs/sapien_env.py`` @ v3.0.1):

=========================  ==================================================
BaseEnv method             Agent dereference / handling here
=========================  ==================================================
``get_state_dict``         ``self.agent.get_controller_state()`` -> overridden
``_get_obs_agent``         ``self.agent.get_proprioception()``  -> overridden
``_step_action(None)``     guarded by ``if self.agent is not None`` (safe)
``reset``                  guarded (safe); ``__init__`` action space guarded
``control_mode``           dereferences agent -> overridden to return None
``_get_obs_with_sensor_data`` / ``_after_reconfigure``  guarded (safe)
``evaluate`` / ``get_info``  no agent use (overridden to report q only)
=========================  ==================================================
"""
from __future__ import annotations

import functools

import numpy as np

from ..geometry.se3 import quat_from_matrix


@functools.lru_cache(maxsize=1)
def _env_class():
    """Define the BaseEnv subclass lazily (no SAPIEN import at package import)."""
    import sapien
    import torch
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.structs.types import SimConfig

    class ArticulatedTwinEnv(BaseEnv):
        SUPPORTED_ROBOTS = ["none"]
        SUPPORTED_REWARD_MODES = ("none",)
        SUPPORTED_OBS_MODES = ("none", "state", "state_dict")

        def __init__(self, *, twin_config, keep_visuals: bool = False, **kwargs):
            # Must be set before BaseEnv.__init__ (which calls _load_scene).
            self._twin_config = twin_config
            self._keep_visuals = keep_visuals
            self.twin = None
            super().__init__(robot_uids="none", **kwargs)

        @property
        def _default_sim_config(self):
            s = self._twin_config.sim
            return SimConfig(sim_freq=s.sim_freq, control_freq=s.control_freq)

        @property
        def _default_sensor_configs(self):
            return []

        @property
        def _default_human_render_camera_configs(self):
            return []

        @property
        def control_mode(self):
            return None

        def _load_agent(self, options, initial_agent_poses=None, build_separate=False):
            super()._load_agent(options, initial_agent_poses, build_separate)
            assert self.agent is None, "ArticulatedTwinEnv must not create an agent"

        def _load_scene(self, options):
            cfg = self._twin_config
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = bool(cfg.fixed_base)
            loader.name = "parttwin_object"
            parsed = loader.parse(str(cfg.urdf))
            builders = parsed["articulation_builders"]
            if len(builders) != 1 or parsed["actor_builders"]:
                raise ValueError(
                    f"{cfg.urdf}: expected exactly one articulation and no free rigid "
                    f"bodies, got {len(builders)} articulation(s), "
                    f"{len(parsed['actor_builders'])} actor(s)")
            builder = builders[0]
            if not self._keep_visuals:
                for lb in builder.link_builders:
                    lb.visual_records.clear()
            T = cfg.T_world_root
            builder.initial_pose = sapien.Pose(p=T[:3, 3].tolist(), q=quat_from_matrix(T[:3, :3]).tolist())
            self.twin = builder.build(name="parttwin_object", fix_root_link=cfg.fixed_base)
            # Drive properties must be set before GPU initialization.
            s = cfg.sim
            for j in self.twin.get_active_joints():
                j.set_drive_properties(s.drive_stiffness, s.drive_damping, s.drive_force_limit)

        def _initialize_episode(self, env_idx, options):
            q0 = initial_qpos(self.twin, self._twin_config.initial_q)
            self.twin.set_qpos(q0)
            self.twin.set_qvel(torch.zeros_like(q0))
            # `joints=` is ManiSkill's recommended form on both CPU and GPU
            # (cached by joint identity in get_joint_target_indices).
            self.twin.set_joint_drive_targets(q0, joints=self.twin.get_active_joints())

        # --- robot-free overrides ------------------------------------------------
        def get_state_dict(self):
            # BaseEnv.get_state_dict dereferences self.agent (None here).
            return self.scene.get_sim_state()

        def _get_obs_agent(self):
            return dict()

        def _get_obs_extra(self, info):
            return dict(qpos=self.twin.get_qpos(), qvel=self.twin.get_qvel())

        def evaluate(self):
            return dict()

    return ArticulatedTwinEnv


def initial_qpos(articulation, initial_q: dict):
    """q0 per active joint: configured value, else 0 clipped into the limits."""
    import torch

    names = [j.name for j in articulation.get_active_joints()]
    unknown = set(initial_q) - set(names)
    if unknown:
        raise ValueError(f"initial_q has non-active joints {sorted(unknown)}; active: {names}")
    lim = articulation.get_qlimits()  # [B, D, 2]
    q = torch.zeros(lim.shape[:-1], device=lim.device, dtype=lim.dtype)
    for i, n in enumerate(names):
        if n in initial_q:
            q[:, i] = float(initial_q[n])
    lo, hi = lim[..., 0], lim[..., 1]
    finite = torch.isfinite(lo) & torch.isfinite(hi)
    q = torch.where(finite, torch.maximum(torch.minimum(q, hi), lo), q)
    return q


def create_env(cfg, sim_backend: str | None = None, obs_mode: str = "state_dict", keep_visuals: bool = False):
    """Construct the robot-free scene (num_envs=1). Physics only, no SAPIEN rendering."""
    backend = sim_backend or cfg.sim.backend
    if backend not in ("physx_cpu", "physx_cuda"):
        raise ValueError("sim_backend must be 'physx_cpu' or 'physx_cuda'")
    cls = _env_class()
    return cls(twin_config=cfg, keep_visuals=keep_visuals, num_envs=1, sim_backend=backend,
               render_backend="none", obs_mode=obs_mode, reward_mode="none", render_mode=None)


# Backwards-compatible name used by the pre-refactor scripts.
def create_articulated_scene(cfg, sim_backend="physx_cpu"):
    return create_env(cfg, sim_backend)


def root_pose_error(env, cfg) -> float:
    """Translation+rotation residual between configured and actual root pose."""
    from ..geometry.se3 import rotation_angle_between, world_from_maniskill_pose

    T = world_from_maniskill_pose(env.twin.root.pose)
    return float(np.linalg.norm(T[:3, 3] - cfg.T_world_root[:3, 3])
                 + rotation_angle_between(T[:3, :3], cfg.T_world_root[:3, :3]))
