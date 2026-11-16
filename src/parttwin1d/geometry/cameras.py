"""Pinhole camera math, OpenCV optical convention (+X right, +Y down, +Z forward).

``T_world_camera`` is camera-to-world (c2w). gsplat expects
``viewmats = inverse(T_world_camera)`` (w2c) and ``Ks`` [C, 3, 3].

gsplat 1.5.3's projection kernels read only fx=K[0,0], fy=K[1,1], cx=K[0,2],
cy=K[1,2] (see ``gsplat/cuda/csrc/ProjectionEWA3DGS*.cu``); a non-zero skew
K[0,1] would be silently ignored, so it is rejected here instead.
Lens distortion is not part of this contract (undistort images upstream).
"""
from __future__ import annotations

import math

import numpy as np

from .se3 import FrameError, invert, transform_points, validate_se3


class CameraError(ValueError):
    """Invalid intrinsics / camera specification."""


def validate_K(K, name: str = "K", width: int | None = None, height: int | None = None) -> np.ndarray:
    K = np.array(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise CameraError(f"{name}: expected 3x3, got {K.shape}")
    if not np.isfinite(K).all():
        raise CameraError(f"{name}: non-finite entries")
    if not np.allclose(K[2], [0.0, 0.0, 1.0]) or K[1, 0] != 0.0:
        raise CameraError(f"{name}: must be upper-triangular with last row [0,0,1]")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise CameraError(f"{name}: fx and fy must be positive (got {K[0,0]}, {K[1,1]})")
    if K[0, 1] != 0.0:
        raise CameraError(
            f"{name}: non-zero skew K[0,1]={K[0,1]} is not supported by the gsplat "
            "1.5.3 pinhole kernel (it would be silently ignored); rectify the camera"
        )
    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise CameraError(f"{name}: width/height must be positive")
        if not (-width <= K[0, 2] <= 2 * width and -height <= K[1, 2] <= 2 * height):
            raise CameraError(f"{name}: principal point ({K[0,2]}, {K[1,2]}) implausible for {width}x{height}")
    return K


def intrinsics_from_vertical_fov(fov_y: float, width: int, height: int) -> np.ndarray:
    """Virtual pinhole K for a GUI orbit camera (square pixels, centered pp).

    Identical to ``nerfview.CameraState.get_K`` at the pinned commit.
    """
    f = height / 2.0 / math.tan(fov_y / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def vertical_fov_from_K(K, height: int) -> float:
    return 2.0 * math.atan(height / (2.0 * float(K[1, 1])))


def scale_intrinsics(K, sx: float, sy: float) -> np.ndarray:
    """K for an image resized by (sx, sy) with pixel-corner convention
    (u' = sx*u): fx*sx, fy*sy, cx*sx, cy*sy."""
    K = np.array(K, dtype=np.float64)
    S = np.diag([sx, sy, 1.0])
    return S @ K


def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV-optical camera-to-world T_world_camera looking from eye to target."""
    eye, target, up = (np.asarray(a, np.float64) for a in (eye, target, up))
    f = target - eye
    if np.linalg.norm(f) < 1e-10:
        raise FrameError("camera eye and target coincide")
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-8:
        raise FrameError("camera up vector is parallel to the viewing direction")
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, f, eye
    return validate_se3(T, "look_at c2w")


def optical_view_from_look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera (viewmat) for the same look-at camera."""
    return invert(look_at_c2w(eye, target, up))


def project_points(K, T_world_camera, xyz_world):
    """Project world points; returns (uv [N,2], z [N]) with z the optical depth.

    Points with z <= 0 are behind the camera; their uv are still returned.
    """
    K = np.asarray(K, np.float64)
    pc = transform_points(invert(T_world_camera), xyz_world)
    z = pc[..., 2]
    uv = (pc[..., :2] / z[..., None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z
