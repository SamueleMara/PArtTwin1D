"""Explicit frame conventions for the simulator/renderer boundary.

T_A_B maps column-vector coordinates in frame B to frame A.
World, URDF-link, Gaussian-PLY, and OpenCV-optical-camera frames are distinct.
All quaternions are (w,x,y,z). No implicit recentering/rescaling.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def se3(matrix, name="transform") -> np.ndarray:
    t = np.asarray(matrix, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f"{name}: expected finite 4x4 matrix")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name}: invalid homogeneous bottom row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError(f"{name}: rotation must be proper and unscaled")
    return t


def invert(t):
    t = se3(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def transform_points(t, xyz):
    t = se3(t)
    x = np.asarray(xyz, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError("points must have final dimension 3")
    return x @ t[:3, :3].T + t[:3, 3]


def matrix_from_pose(p, q_wxyz):
    p = np.asarray(p, dtype=np.float64).reshape(3)
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    if np.linalg.norm(q) < 1e-12:
        raise ValueError("zero-norm quaternion")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]] / np.linalg.norm(q)).as_matrix()
    t[:3, 3] = p
    return t


def quat_from_matrix(r):
    q_xyzw = Rotation.from_matrix(np.asarray(r)).as_quat()
    return q_xyzw[[3, 0, 1, 2]]


def quat_multiply(a, b):
    """Torch batched Hamilton product, scalar-first quaternions."""
    import torch
    wa, xa, ya, za = a.unbind(-1)
    wb, xb, yb, zb = b.unbind(-1)
    return torch.stack((
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb,
    ), dim=-1)


def optical_view_from_look_at(eye, target, up=(0., 0., 1.)):
    """World-to-camera matrix, OpenCV optical axes (+x right, +y down, +z forward)."""
    eye, target, up = map(lambda a: np.asarray(a, np.float64), (eye, target, up))
    f = target - eye
    if np.linalg.norm(f) < 1e-10:
        raise ValueError("camera eye and target coincide")
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("camera up parallel to forward")
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    r = np.stack((right, down, f))
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = -r @ eye
    return se3(t)


def world_from_mani_skill_pose(pose):
    """Batched (B=1) ManiSkill Link.pose into a single NumPy T_world_link."""
    p, q = pose.p, pose.q
    if hasattr(p, "detach"):
        p, q = p.detach().cpu().numpy(), q.detach().cpu().numpy()
    p = np.asarray(p)
    q = np.asarray(q)
    if p.size != 3 or q.size != 4:
        raise ValueError(f"Expected one environment; got pose shapes {p.shape}, {q.shape}")
    return matrix_from_pose(p.reshape(3), q.reshape(4))
