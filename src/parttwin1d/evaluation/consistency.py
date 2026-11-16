"""Geometry / joint / render invariants used by tests, the render CLI and reports."""
from __future__ import annotations

import numpy as np

from ..geometry.se3 import axis_angle_matrix, invert, rotation_angle_between


def expected_child_pose(T_W_child_q0, joint, q: float, q0: float) -> np.ndarray:
    """Analytic child-link pose for a single revolute/prismatic joint.

    URDF: the child link frame coincides with the joint frame, so the motion
    from q0 to q is a rotation/translation about/along ``joint.axis`` expressed
    in the child frame at q0:  T_W_C(q) = T_W_C(q0) @ M(axis, q - q0).
    """
    dq = q - q0
    if joint.type in ("revolute", "continuous"):
        M = axis_angle_matrix(joint.axis, dq)
    elif joint.type == "prismatic":
        M = np.eye(4)
        M[:3, 3] = np.asarray(joint.axis) * dq
    else:
        raise ValueError(f"unsupported joint type {joint.type}")
    return T_W_child_q0 @ M


def pose_error(Ta, Tb) -> tuple[float, float]:
    """(translation error m, rotation error rad)."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), rotation_angle_between(Ta[:3, :3], Tb[:3, :3])


def static_link_invariance(snapshots, link: str) -> tuple[float, float]:
    """Max pose deviation of a link across snapshots (should be ~0 for fixed links)."""
    T0 = snapshots[0].T_world_link[link]
    errs = [pose_error(T0, s.T_world_link[link]) for s in snapshots[1:]]
    return (max(e[0] for e in errs), max(e[1] for e in errs)) if errs else (0.0, 0.0)


def hinge_pivot_residual(T_W_child, joint, T_W_child_q0) -> float:
    """Distance the joint pivot point moved (should be ~0 for a revolute joint)."""
    p_q0 = T_W_child_q0[:3, 3]
    p_q = T_W_child[:3, 3]
    return float(np.linalg.norm(p_q - p_q0))


def relative_link_motion(T_W_link_a, T_W_link_b) -> np.ndarray:
    return invert(T_W_link_a) @ T_W_link_b


def image_change(rgb_a, rgb_b) -> float:
    """Mean absolute RGB difference in [0,1] units."""
    a = rgb_a.float() if hasattr(rgb_a, "float") else np.asarray(rgb_a, np.float32)
    b = rgb_b.float() if hasattr(rgb_b, "float") else np.asarray(rgb_b, np.float32)
    d = (a - b).abs().mean() if hasattr(a, "abs") else np.abs(a - b).mean()
    return float(d)
