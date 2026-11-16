"""Native ManiSkill articulated URDF scene, without fake robots or mesh appearance.

This is a *physical object* scene. It is deliberately not a policy task.
For robot PPO, compose the same renderer with an ordinary robot-equipped ManiSkill
BaseEnv and use its live `Articulation` link poses; do not invent a new physics path.
"""
from __future__ import annotations

from pathlib import Path
from .geometry import quat_from_matrix


def build_env_type(urdf: Path, *, fixed_base: bool, T_world_root):
    """Deferred import lets geometry and asset tests run without SAPIEN installed."""
    import torch
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.structs.types import SimConfig

    class ArticulatedTwinEnv(BaseEnv):
        SUPPORTED_ROBOTS = ["none"]
        SUPPORTED_REWARD_MODES = ("none",)

        def __init__(self, **kwargs):
            super().__init__(robot_uids="none", **kwargs)

        @property
        def _default_sim_config(self):
            return SimConfig(sim_freq=200, control_freq=20)

        @property
        def _default_sensor_configs(self):
            return []

        @property
        def _default_human_render_camera_configs(self):
            return []

        def _load_scene(self, options):
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = bool(fixed_base)
            loader.name = "parttwin_object"
            # Explicit builder-based load to set root pose before PhysX initializes.
            parsed = loader.parse(str(urdf))
            builders = parsed["articulation_builders"]
            if len(builders) != 1 or parsed["actor_builders"]:
                raise ValueError("Expected exactly one articulated URDF (not several rigid bodies)")
            builder = builders[0]
            # Visual appearance belongs exclusively to gsplat. Keep PhysX
            # collision/inertia records, but do not instantiate URDF meshes as
            # native SAPIEN render components when render_backend="none".
            for link_builder in builder.link_builders:
                link_builder.visual_records.clear()
            q = quat_from_matrix(T_world_root[:3, :3]).tolist()
            builder.initial_pose = sapien.Pose(p=T_world_root[:3, 3].tolist(), q=q)
            self.twin = builder.build(name="parttwin_object", fix_root_link=fixed_base)

        def get_state_dict(self):
            # ManiSkill 3.0.1's default unconditionally accesses an agent's
            # controller; a robot-free scene must return only physical state.
            return self.scene.get_sim_state()

        def _get_obs_agent(self):
            return {}

        def _get_obs_extra(self, info):
            return {"qpos": self.twin.qpos}

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
            q = self.twin.qpos
            limits = self.twin.qlimits  # [B, DoF, 2]
            # Start at zero when valid; otherwise nearest joint-limit endpoint.
            zero = torch.zeros_like(q)
            if q.numel():
                zero = torch.maximum(torch.minimum(zero, limits[..., 1]), limits[..., 0])
            self.twin.set_qpos(zero)
            self.twin.set_qvel(torch.zeros_like(q))

        def evaluate(self):
            return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    return ArticulatedTwinEnv


def create_articulated_scene(cfg, sim_backend="physx_cpu"):
    """Construct a single real PhysX articulation; disable SAPIEN graphics only.

    The Gaussian renderer owns the camera. Render backend 'none' intentionally
    avoids Vulkan setup for this first stage, not physics or contact simulation.
    """
    if sim_backend not in ("physx_cpu", "physx_cuda"):
        raise ValueError("sim_backend must be 'physx_cpu' or 'physx_cuda'")
    cls = build_env_type(cfg.urdf, fixed_base=cfg.root_fixed, T_world_root=cfg.initial_root_pose)
    return cls(num_envs=1, sim_backend=sim_backend, render_backend="none",
               obs_mode="none", reward_mode="none", render_mode=None)


def set_joint_qpos(env, joint_name: str, q: float):
    """Set an articulated joint by name; update actual PhysX link kinematics.

    This is an offline pose sweep helper, NOT a dynamics/actuation model.
    For physical trajectories, step ManiSkill with forces or joint drives instead.
    """
    import torch
    art = env.twin
    names = [joint.name for joint in art.get_active_joints()]
    if joint_name not in names:
        raise ValueError(f"Joint {joint_name!r} not found. Available active joints: {names}")
    index = names.index(joint_name)
    qlimits = art.qlimits
    lower = float(qlimits[0, index, 0])
    upper = float(qlimits[0, index, 1])
    if q < lower-1e-5 or q > upper+1e-5:
        raise ValueError(f"Joint {joint_name} value {q} outside URDF limits [{lower}, {upper}]")
    v = art.get_qpos().clone()
    v[:, index] = q
    art.set_qpos(v)
    art.set_qvel(torch.zeros_like(v))
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
    return {n: float(x) for n, x in zip(names, art.get_qpos()[0].detach().cpu().tolist())}
