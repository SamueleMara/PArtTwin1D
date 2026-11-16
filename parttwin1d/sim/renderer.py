"""Link-pose-driven 3DGS observation renderer backed by the existing gsplat CUDA kernels.

No new rasterizer. No meshes substituted for appearance. Supports every moving
URDF link, exact intrinsics, OpenCV camera extrinsics, alpha and expected depth.
"""
from __future__ import annotations

import numpy as np
from .assets import AssetConfig, load_graphdeco_ply
from .geometry import invert, quat_from_matrix, quat_multiply, transform_points, world_from_mani_skill_pose


class ArticulatedGaussianRenderer:
    """Reusable object-level renderer; independent of a particular ManiSkill task.

    Gaussian geometry is immutable; new world transforms are derived from live
    SAPIEN link poses on every call. `render_tensor` keeps output on the GPU
    for future PPO observations, `render_numpy` is an explicit export boundary.
    """

    def __init__(self, config: AssetConfig, device="cuda", max_gaussians_per_link=0):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat rasterizer needs a CUDA-enabled PyTorch environment")
        self.device = torch.device(device)
        self.config = config
        self.parts = []
        for spec in config.links:
            data = load_graphdeco_ply(spec.path, device=device, limit=max_gaussians_per_link)
            degree = data["degree"] if config.sh_degree == -1 else min(config.sh_degree, data["degree"])
            self.parts.append((spec, data, degree))
        self.gaussian_count = sum(part[1]["count"] for part in self.parts)

    @staticmethod
    def validate_links(articulation, specs):
        names = set(articulation.links_map)
        absent = [s.link for s in specs if s.link not in names]
        if absent:
            raise ValueError(f"PLY link bindings missing in URDF: {absent}; available={sorted(names)}")

    @staticmethod
    def link_world_matrices(articulation, specs):
        ArticulatedGaussianRenderer.validate_links(articulation, specs)
        result = {}
        for spec in specs:
            result[spec.link] = world_from_mani_skill_pose(articulation.links_map[spec.link].pose)
        return result

    @staticmethod
    def local_camera_direction(means_local, T_world_ply, camera_world_position):
        """Camera -> Gaussian in canonical PLY frame, matching gsplat SH sign."""
        import torch
        cam_local = transform_points(invert(T_world_ply), np.asarray(camera_world_position)[None])[0]
        cam = torch.as_tensor(cam_local, device=means_local.device, dtype=means_local.dtype)
        return torch.nn.functional.normalize(means_local - cam[None], dim=-1, eps=1e-8)

    def _world_params(self, link_world, camera_world):
        import torch
        # Import the CUDA SH implementation only when rendering; degree 0 stays cheap.
        from gsplat.cuda._wrapper import spherical_harmonics
        means, rotations, scales, opacities, rgb_colors = [], [], [], [], []
        part_poses = {}
        for spec, data, degree in self.parts:
            T_world_ply = link_world[spec.link] @ spec.T_link_ply
            part_poses[spec.link] = T_world_ply
            R = torch.tensor(T_world_ply[:3, :3], device=self.device, dtype=torch.float32)
            t = torch.tensor(T_world_ply[:3, 3], device=self.device, dtype=torch.float32)
            rot_q = torch.as_tensor(quat_from_matrix(T_world_ply[:3, :3]), device=self.device, dtype=torch.float32)
            m = data["means"] @ R.T + t
            q = torch.nn.functional.normalize(
                quat_multiply(rot_q[None].expand_as(data["quats"]), data["quats"]), dim=-1)
            # Evaluate view-dependent SH in the original PLY coordinate system:
            # otherwise rotating a link would incorrectly rotate its appearance.
            if degree == 0:
                c = data["sh"][:, 0, :] * 0.28209479177387814 + 0.5
            else:
                dirs = self.local_camera_direction(data["means"], T_world_ply, camera_world)
                c = spherical_harmonics(degree, dirs.contiguous(), data["sh"].contiguous()) + 0.5
            means.append(m)
            rotations.append(q)
            scales.append(data["scales"])
            opacities.append(data["opacities"])
            rgb_colors.append(torch.clamp_min(c, 0))
        return (torch.cat(means), torch.cat(rotations), torch.cat(scales),
                torch.cat(opacities), torch.cat(rgb_colors), part_poses)

    def render_tensor(self, articulation, camera):
        """Return GPU tensors (RGB HWC float [0,1], alpha HW, depth HW meters).

        Depth is *expected optical-z depth* from gsplat RGB+ED, with NaN on
        pixels alpha < 0.01; it's not a hard mesh depth or collision surface.
        """
        import torch
        from gsplat.rendering import rasterization
        specs = [x[0] for x in self.parts]
        link_world = self.link_world_matrices(articulation, specs)
        camera_world = camera.T_world_camera[:3, 3]
        with torch.no_grad():
            mu, q, s, alpha_g, color, part_poses = self._world_params(link_world, camera_world)
            view = torch.as_tensor(invert(camera.T_world_camera), dtype=torch.float32, device=self.device)[None]
            K = torch.as_tensor(camera.K, dtype=torch.float32, device=self.device)[None]
            bg = torch.as_tensor([self.config.render_background], dtype=torch.float32, device=self.device)
            rgbd, alpha, _ = rasterization(
                means=mu, quats=q, scales=s, opacities=alpha_g, colors=color,
                viewmats=view, Ks=K, width=camera.width, height=camera.height,
                sh_degree=None, render_mode="RGB+ED", backgrounds=bg, packed=True)
            rgb = rgbd[0, :, :, :3].clamp(0., 1.)
            a = alpha[0, :, :, 0]
            depth = rgbd[0, :, :, 3].masked_fill(a < 0.01, float("nan"))
            return {"rgb": rgb, "alpha": a, "depth_m": depth, "T_world_ply": part_poses}

    def render_numpy(self, articulation, camera):
        """CPU conversion for file export, never used in policy observation path."""
        d = self.render_tensor(articulation, camera)
        return {
            "rgb": (d["rgb"] * 255).to(dtype=__import__("torch").uint8).cpu().numpy(),
            "alpha": d["alpha"].cpu().numpy(),
            "depth_m": d["depth_m"].cpu().numpy(),
            "T_world_ply": {k: v.tolist() for k, v in d["T_world_ply"].items()},
        }
