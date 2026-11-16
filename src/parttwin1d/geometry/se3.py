"""Single SE(3) / quaternion implementation for the whole project.

Notation (see docs/coordinate_frames.md)
----------------------------------------
``T_A_B`` (written T_{A<-B} in the docs) maps homogeneous column vectors in
frame B to frame A::

    [x_A; 1] = T_A_B @ [x_B; 1]          T_A_C = T_A_B @ T_B_C

Frames: ``W`` simulator world, ``L_i`` URDF link frame, ``P_i`` canonical PLY
frame of link i, ``C`` OpenCV optical camera (+X right, +Y down, +Z forward).

Quaternions are **scalar-first (w, x, y, z)**, Hamilton product, active
rotation: ``R(q1 ⊗ q2) = R(q1) @ R(q2)``. This matches gsplat's
``quats`` convention (``gsplat.cuda._torch_impl._quat_to_rotmat``).

No implicit recentering, rescaling or axis swapping happens anywhere.
NumPy functions operate in float64 for configuration/validation; the Torch
functions are batched and device-agnostic for the render hot path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:  # pragma: no cover
    import torch

ROT_ORTHO_TOL = 1e-5
DET_TOL = 1e-5


class FrameError(ValueError):
    """Invalid rigid transform / frame specification."""


# --------------------------------------------------------------------------- #
# NumPy (float64) — configuration, validation, export
# --------------------------------------------------------------------------- #
def validate_se3(matrix, name: str = "transform") -> np.ndarray:
    """Return a float64 copy of a validated proper rigid 4x4 transform.

    Rejects: wrong shape, non-finite values, bad homogeneous row, non-orthonormal
    rotation (i.e. scale/shear), and reflections (det(R) = -1).
    """
    try:
        t = np.array(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FrameError(f"{name}: not numeric ({exc})") from exc
    if t.shape != (4, 4):
        raise FrameError(f"{name}: expected 4x4 matrix, got shape {t.shape}")
    if not np.isfinite(t).all():
        raise FrameError(f"{name}: contains non-finite values")
    if not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise FrameError(f"{name}: bottom row must be [0, 0, 0, 1], got {t[3].tolist()}")
    r = t[:3, :3]
    det = float(np.linalg.det(r))
    if det < 0:
        raise FrameError(f"{name}: reflection (det(R)={det:.6f}); a right-handed rotation is required")
    if not np.allclose(r.T @ r, np.eye(3), atol=ROT_ORTHO_TOL) or abs(det - 1.0) > DET_TOL:
        raise FrameError(
            f"{name}: rotation block is not orthonormal (det={det:.6f}); scale/shear "
            "must be removed explicitly, not hidden in T"
        )
    return t


# Backwards-compatible short name used by the pre-refactor code.
se3 = validate_se3


def invert(t) -> np.ndarray:
    """Closed-form inverse of a rigid transform: T_B_A from T_A_B."""
    t = validate_se3(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def compose(*transforms) -> np.ndarray:
    """compose(T_A_B, T_B_C, ...) -> T_A_C (left-to-right matrix product)."""
    if not transforms:
        return np.eye(4)
    out = validate_se3(transforms[0])
    for t in transforms[1:]:
        out = out @ validate_se3(t)
    return out


def transform_points(t, xyz) -> np.ndarray:
    """x_A = R_AB x_B + t_AB for points [..., 3]."""
    t = validate_se3(t)
    x = np.asarray(xyz, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError(f"points must have final dimension 3, got {x.shape}")
    return x @ t[:3, :3].T + t[:3, 3]


def rotate_covariance(R, cov) -> np.ndarray:
    """Sigma_A = R_AB Sigma_B R_AB^T for covariances [..., 3, 3]."""
    R = np.asarray(R, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    return R @ cov @ np.swapaxes(R, -1, -2)


def quat_wxyz_to_matrix(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise FrameError("zero-norm quaternion")
    q = q / n
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def matrix_from_pose(p, q_wxyz) -> np.ndarray:
    """T from position p and scalar-first quaternion (SAPIEN/ManiSkill pose)."""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    t = np.eye(4)
    t[:3, :3] = quat_wxyz_to_matrix(q)
    t[:3, 3] = p
    return t


def quat_from_matrix(r) -> np.ndarray:
    """Scalar-first quaternion with w >= 0 from a rotation matrix."""
    q = Rotation.from_matrix(np.asarray(r, dtype=np.float64)).as_quat()[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0, -q, q)


def rotation_angle_between(Ra, Rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def axis_angle_matrix(axis, angle: float, pivot=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rigid transform rotating by ``angle`` about ``axis`` through ``pivot``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    p = np.asarray(pivot, dtype=np.float64)
    t[:3, 3] = p - t[:3, :3] @ p
    return t


def world_from_maniskill_pose(pose, env_index: int = 0) -> np.ndarray:
    """ManiSkill batched ``Pose`` (p [B,3], q [B,4] wxyz) -> NumPy T_world_x.

    Explicit CPU boundary: used for the single-scene UI/export path only.
    """
    p, q = pose.p, pose.q
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
        q = q.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    if env_index >= len(p):
        raise IndexError(f"env_index {env_index} out of range for batch {len(p)}")
    return matrix_from_pose(p[env_index], q[env_index])


# --------------------------------------------------------------------------- #
# Torch (batched) — render hot path, stays on device
# --------------------------------------------------------------------------- #
def quat_multiply(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Batched Hamilton product a ⊗ b of scalar-first quaternions [..., 4]."""
    import torch

    wa, xa, ya, za = a.unbind(-1)
    wb, xb, yb, zb = b.unbind(-1)
    return torch.stack(
        (
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ),
        dim=-1,
    )


def quat_to_matrix_torch(q: "torch.Tensor") -> "torch.Tensor":
    """Scalar-first quaternions [..., 4] -> rotation matrices [..., 3, 3]."""
    import torch

    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat_torch(R: "torch.Tensor") -> "torch.Tensor":
    """Rotation matrices [..., 3, 3] -> scalar-first unit quaternions, w >= 0.

    Shepperd's method, numerically stable for all rotations.
    """
    import torch

    m = R
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    cand = torch.stack(
        (
            torch.stack((1 + tr, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20), -1),
            torch.stack((m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21), -1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), -1),
        ),
        dim=-2,
    )  # [..., 4, 4]; row k is 4*q_k*q (up to scale)
    diag = torch.stack((1 + tr, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    idx = diag.argmax(-1, keepdim=True)
    q = torch.gather(cand, -2, idx[..., None].expand(*idx.shape, 4)).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def invert_torch(T: "torch.Tensor") -> "torch.Tensor":
    """Closed-form batched rigid inverse [..., 4, 4]."""
    import torch

    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def pose_tensor_to_matrix(p: "torch.Tensor", q_wxyz: "torch.Tensor") -> "torch.Tensor":
    """Batched (p [...,3], q [...,4]) -> T [...,4,4] on the same device (no CPU)."""
    import torch

    T = torch.zeros(p.shape[:-1] + (4, 4), dtype=p.dtype, device=p.device)
    T[..., :3, :3] = quat_to_matrix_torch(q_wxyz)
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T
