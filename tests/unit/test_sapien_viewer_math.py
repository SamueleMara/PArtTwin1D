"""SAPIEN viewer integration pieces that run without a window: shader patch and camera math."""
from types import SimpleNamespace

import numpy as np
import pytest

from parttwin1d.geometry.cameras import look_at_c2w
from parttwin1d.viewer.sapien_shader import ShaderPatchError, build_shader_pack, patch_composite


def test_shader_pack_generated_from_installed_sapien(tmp_path):
    pack = build_shader_pack(tmp_path)
    frag = (pack / "composite0.frag").read_text()
    assert frag.count("samplerGaussianColor") == 2 and frag.count("samplerGaussianDepth") == 2
    assert (pack.parent / "common").is_dir() and (pack / "gbuffer.frag").exists()
    assert build_shader_pack(tmp_path) == pack  # idempotent / cached


def test_shader_patch_fails_loudly_on_unknown_shader():
    with pytest.raises(ShaderPatchError):
        patch_composite("void main() {}")


def test_window_camera_conversion_roundtrip():
    """OpenGL model/projection of the SAPIEN window -> OpenCV K and c2w (pure math)."""
    from parttwin1d.viewer.maniskill_viewer import GL_TO_CV, sapien_pose_from_c2w, window_render_camera

    W, H, fovy = 800, 600, 0.9
    T_cv = look_at_c2w([1.0, -1.2, 0.8], [0.2, 0.1, 0.3])
    f = 1 / np.tan(fovy / 2)
    P = np.array([[f * H / W, 0, 0, 0], [0, -f, 0, 0], [0, 0, -1, -0.01], [0, 0, -1, 0]])
    win = SimpleNamespace(get_picture_size=lambda name: (W, H), size=(W, H),
                          get_camera_projection_matrix=lambda: P,
                          get_camera_model_matrix=lambda: T_cv @ np.linalg.inv(GL_TO_CV))
    cam = window_render_camera(win)
    np.testing.assert_allclose(cam.T_world_camera, T_cv, atol=1e-12)
    np.testing.assert_allclose(cam.K, [[H / 2 * f, 0, W / 2], [0, H / 2 * f, H / 2], [0, 0, 1]], atol=1e-9)
    half = window_render_camera(win, 0.5)
    assert (half.width, half.height) == (400, 300) and half.K[0, 0] == pytest.approx(cam.K[0, 0] / 2)
    # SAPIEN camera pose convention (x forward, y left, z up)
    pose = sapien_pose_from_c2w(T_cv)
    from parttwin1d.geometry.se3 import matrix_from_pose

    R_s = matrix_from_pose(pose.p, pose.q)[:3, :3]
    np.testing.assert_allclose(R_s[:, 0], T_cv[:3, 2], atol=1e-6)   # forward = optical +Z
    np.testing.assert_allclose(R_s[:, 2], -T_cv[:3, 1], atol=1e-6)  # up = -optical Y
