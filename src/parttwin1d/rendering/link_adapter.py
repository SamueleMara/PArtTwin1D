"""Exact per-link rigid transport of canonical Gaussians.

For link i with ``T_W_Pi = T_W_Li(q) @ T_Li_Pi`` (R, t):

    mu_W    = R mu_P + t
    q_W     = q(R) ⊗ q_P            (=> R(q_W) = R R(q_P), Sigma_W = R Sigma_P R^T)
    scales  unchanged
    SH      evaluated in the canonical P frame with the camera->Gaussian direction
            expressed in P:  d_P = R^T (mu_W - c_W) = mu_P - R^T (c_W - t)

The last line is what keeps view-dependent colour attached to the part when
it rotates; gsplat's own SH path would evaluate d_W in the world basis, which
is only correct for R = I. Results are written into persistent output buffers
from the immutable canonical inputs on every call (no incremental drift).
"""
from __future__ import annotations

from ..geometry.se3 import matrix_to_quat_torch, quat_multiply

SH_C0 = 0.28209479177387814


class LinkAdapter:
    def __init__(self, scene):
        import torch

        self.scene = scene
        dev = scene.means.device
        self.means_w = torch.empty_like(scene.means)
        self.quats_w = torch.empty_like(scene.quats)
        self.T_world_ply = torch.eye(4, device=dev).repeat(len(scene.link_names), 1, 1)

    def update(self, T_world_link):
        """T_world_link: torch [L,4,4] (scene link order). Returns (means_w, quats_w)."""
        import torch

        sc = self.scene
        T_wp = T_world_link.to(sc.T_link_ply) @ sc.T_link_ply  # [L,4,4]
        q_wp = matrix_to_quat_torch(T_wp[:, :3, :3])
        for i in range(len(sc.link_names)):
            s = sc.link_slice(i)
            R, t = T_wp[i, :3, :3], T_wp[i, :3, 3]
            torch.addmm(t, sc.means[s], R.T, out=self.means_w[s])
            self.quats_w[s] = quat_multiply(q_wp[i].expand(s.stop - s.start, 4), sc.quats[s])
        self.T_world_ply = T_wp
        return self.means_w, self.quats_w

    def colors(self, camera_positions, degree: int):
        """View-dependent RGB [C,N,3] for camera centres [C,3] (world), canonical-frame SH."""
        import torch

        sc = self.scene
        if degree == 0:
            c = torch.clamp_min(sc.sh[:, 0, :] * SH_C0 + 0.5, 0.0)
            return c.unsqueeze(0).expand(camera_positions.shape[0], -1, -1)
        from gsplat.cuda._wrapper import spherical_harmonics

        T = self.T_world_ply  # [L,4,4]
        R, t = T[:, :3, :3], T[:, :3, 3]
        # camera centre in each link's PLY frame: [C,L,3]
        cam_p = torch.einsum("lji,clj->cli", R, camera_positions[:, None, :] - t[None])
        dirs = sc.means[None] - cam_p[:, sc.link_ids]  # [C,N,3]
        C = camera_positions.shape[0]
        coeffs = sc.sh[None].expand(C, -1, -1, -1)
        out = spherical_harmonics(degree, dirs, coeffs)
        return torch.clamp_min(out + 0.5, 0.0)
