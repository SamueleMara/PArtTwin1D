"""Pinhole / OpenCV camera contract (CPU)."""
import numpy as np
import pytest

from parttwin1d.geometry.cameras import (
    CameraError, intrinsics_from_vertical_fov, look_at_c2w, project_points, scale_intrinsics, validate_K,
    vertical_fov_from_K,
)
from parttwin1d.geometry.se3 import invert
from parttwin1d.rendering.cameras import RenderCamera


def test_optical_axes_positive_and_negative():
    T = look_at_c2w([0, -2, 0], [0, 0, 0])  # looking along +Y world, z-up
    np.testing.assert_allclose(T[:3, 2], [0, 1, 0], atol=1e-12)    # +Z optical = forward
    np.testing.assert_allclose(T[:3, 1], [0, 0, -1], atol=1e-12)   # +Y optical = down (world -Z)
    np.testing.assert_allclose(T[:3, 0], [1, 0, 0], atol=1e-12)    # +X optical = right
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    uv, z = project_points(K, T, [[0, 0, 0], [0, -3, 0], [0.1, 0, 0], [0, 0, 0.1]])
    assert z[0] > 0 and z[1] < 0                       # in front / behind
    np.testing.assert_allclose(uv[0], [320, 240])
    assert uv[2, 0] > 320                               # world +X -> image right
    assert uv[3, 1] < 240                               # world up -> image up (smaller v)


def test_exact_K_fx_fy_cx_cy():
    K = np.array([[600.0, 0, 331.5], [0, 590.0, 236.0], [0, 0, 1]])
    T = np.eye(4)
    uv, _ = project_points(K, T, [[0.1, 0.2, 2.0]])
    np.testing.assert_allclose(uv[0], [331.5 + 600 * 0.05, 236 + 590 * 0.1])


def test_skew_and_bad_K_rejected():
    with pytest.raises(CameraError, match="skew"):
        validate_K([[500, 1.0, 320], [0, 500, 240], [0, 0, 1]])
    with pytest.raises(CameraError, match="positive"):
        validate_K([[-500, 0, 320], [0, 500, 240], [0, 0, 1]])
    with pytest.raises(CameraError):
        validate_K([[500, 0, 320], [0, 500, 240], [0, 0, 2]])


def test_virtual_intrinsics_match_nerfview():
    nerfview = pytest.importorskip("nerfview")
    cs = nerfview.CameraState(fov=0.9, aspect=4 / 3, c2w=np.eye(4))
    np.testing.assert_allclose(intrinsics_from_vertical_fov(0.9, 640, 480), cs.get_K((640, 480)))
    assert abs(vertical_fov_from_K(intrinsics_from_vertical_fov(0.9, 640, 480), 480) - 0.9) < 1e-12


def test_resized_camera_scales_K_exactly():
    cam = RenderCamera("c", 640, 480, [[600.0, 0, 331.5], [0, 590.0, 236.0], [0, 0, 1]], np.eye(4))
    half = cam.resized(320, 240)
    np.testing.assert_allclose(half.K, [[300, 0, 165.75], [0, 295, 118], [0, 0, 1]])
    np.testing.assert_allclose(scale_intrinsics(cam.K, 2, 2)[0, 2], 663.0)


def test_viewmat_is_inverse_of_c2w():
    T = look_at_c2w([1, -1, 1], [0, 0, 0])
    cam = RenderCamera("c", 64, 48, [[50.0, 0, 32], [0, 50, 24], [0, 0, 1]], T)
    np.testing.assert_allclose(cam.viewmat, invert(T))
