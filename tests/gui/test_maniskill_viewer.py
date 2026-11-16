"""REAL ManiSkill/SAPIEN viewer tests: Gaussians composited into ManiSkill's own
viewer window (``env._viewer``, sapien.utils.Viewer) — no mocks.

Opt-in:  PARTTWIN_RUN_GUI_TESTS=1 python -m pytest -v tests/gui/test_maniskill_viewer.py
Needs a display (DISPLAY) and Vulkan; opens a desktop window briefly.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from tests.conftest import HINGED, ROOT, requires_gui

pytestmark = [requires_gui, pytest.mark.gui,
              pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                                 reason="needs a desktop display for the SAPIEN viewer window")]
ART = ROOT / "outputs" / "test_artifacts"


def frame(app, n=1):
    for _ in range(n):
        app.step()
        app.env.render_human()


def picture(app, name="Color"):
    return np.asarray(app.window.get_picture(name))


def red_mask(rgb):
    return (rgb[..., 0] > 0.6) & (rgb[..., 1] < 0.4) & (rgb[..., 2] < 0.4)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from parttwin1d.config import load_config
    from parttwin1d.viewer.maniskill_viewer import ManiSkillGaussianViewer

    a = ManiSkillGaussianViewer(load_config(HINGED), width=960, height=720,
                                out_dir=tmp_path_factory.mktemp("msgui"))
    frame(a, 5)
    yield a
    a.close()


def test_uses_maniskill_viewer_and_shader_pack(app):
    from sapien.utils import Viewer

    assert isinstance(app.viewer, Viewer) and app.env._viewer is app.viewer
    assert app.env._viewer_camera_config.shader_config.shader_pack == str(app.shader_pack)
    assert "samplerGaussianColor" in (app.shader_pack / "composite0.frag").read_text()
    assert app.plugin in app.viewer.plugins
    assert app.env.agent is None and app.env.gpu_sim_enabled is False


def test_gaussians_visible_in_composited_frame(app):
    app.show_proxies = False
    frame(app, 2)
    rgb = picture(app)[..., :3]
    assert red_mask(rgb).sum() > 1500, "gsplat door not visible in the SAPIEN viewer frame"
    from PIL import Image

    from parttwin1d.rendering.observations import rgb_uint8

    ART.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8(rgb)).save(ART / "maniskill_viewer_q0.png")


def test_same_camera_same_frame_zero_lag(app):
    """After a camera change, the Gaussian layer of THAT frame used the new SAPIEN camera."""
    from parttwin1d.geometry.cameras import look_at_c2w
    from parttwin1d.viewer.maniskill_viewer import GL_TO_CV, sapien_pose_from_c2w

    c, r = app.object_center_radius()
    for k, d in enumerate(([1.0, -1.0, 0.6], [-0.4, -1.5, 1.0], [1.4, 0.2, 0.8])):
        d = np.array(d) / np.linalg.norm(d)
        app.viewer.set_camera_pose(sapien_pose_from_c2w(look_at_c2w(c + d * 3.0 * r, c)))
        n0 = app.plugin.frames
        frame(app, 1)
        T_window = np.asarray(app.window.get_camera_model_matrix(), np.float64) @ GL_TO_CV
        assert app.plugin.frames == n0 + 1
        np.testing.assert_allclose(app.plugin.last_camera.T_world_camera, T_window, atol=1e-5)


def test_gsplat_depth_matches_sapien_rasterized_proxy(app):
    """Door Gaussians lie inside the door's collision box: gsplat depth (our camera) must
    match SAPIEN's own rasterized depth of the proxy at the same pixels (same camera)."""
    app.free_camera()
    app.show_proxies = True
    frame(app, 2)
    pos = picture(app, "Position")
    sapien_z = -pos[..., 2]
    has_geom = pos[..., 3] < 1.0
    seg = None
    res = app.plugin.last_result
    g_depth = res.depth.cpu().numpy()
    lid = app.renderer.render(app.plugin.last_camera, segmentation=True).link_id.cpu().numpy()
    seg = (lid == app.renderer.link_names.index("door_link")) & res.valid.cpu().numpy() & has_geom
    assert seg.sum() > 1000
    diff = np.abs(g_depth[seg] - sapien_z[seg])
    (ART / "maniskill_viewer_depth_agreement.json").write_text(json.dumps(
        {"pixels": int(seg.sum()), "median_abs_m": float(np.median(diff)), "p90_abs_m": float(np.percentile(diff, 90))}))
    # proxy surface is 1 cm in front of the splat plane (box half-thickness 0.01 m)
    assert np.median(diff) < 0.02, np.median(diff)
    # with opaque proxies shown, the splats inside them are occluded
    assert red_mask(picture(app)[..., :3]).sum() < 200
    app.show_proxies = False
    frame(app, 2)
    assert red_mask(picture(app)[..., :3]).sum() > 1500


def test_joint_motion_updates_layer_and_sapien(app):
    from parttwin1d.evaluation.consistency import expected_child_pose, pose_error

    s0 = app.current_snapshot()
    before = picture(app)[..., :3].copy()
    app.set_joint("door_hinge", 1.2)
    frame(app, 1)
    s1 = app.current_snapshot()
    assert s1.revision > s0.revision and abs(s1.q[0] - 1.2) < 1e-5
    assert app.plugin.last_result.revision == s1.revision  # layer rendered from the new snapshot this frame
    uj = app.driver.urdf.joints["door_hinge"]
    tr, rot = pose_error(expected_child_pose(s0.T_world_link["door_link"], uj, 1.2, float(s0.q[0])),
                         s1.T_world_link["door_link"])
    assert rot < 1e-4 and tr < 1e-5
    after = picture(app)[..., :3]
    assert np.abs(after - before).mean() > 0.002
    from PIL import Image

    from parttwin1d.rendering.observations import rgb_uint8

    Image.fromarray(rgb_uint8(after)).save(ART / "maniskill_viewer_q1.2.png")
    # SAPIEN's own articulation API (what its ArticulationWindow uses) also drives the splats
    import torch

    app.env.twin.set_qpos(torch.tensor([[0.3]]))
    frame(app, 1)
    assert abs(app.plugin.last_result.q["door_hinge"] - 0.3) < 1e-5
    app.reset()
    frame(app, 1)


def test_physics_and_sweep_from_panel_state(app):
    app.mode_index = 1
    app.set_joint("door_hinge", 1.0)  # PD target in dynamics mode
    app.run_physics = True
    t0 = time.time()
    while time.time() - t0 < 1.5:
        frame(app, 1)
    q = app.current_snapshot().q[0]
    assert app.current_snapshot().sim_time > 0.5 and q > 0.5
    app.run_physics = False
    app.mode_index = 0
    app.playing = True
    qs = []
    for _ in range(20):
        frame(app, 1)
        qs.append(float(app.current_snapshot().q[0]))
    app.playing = False
    assert len(set(np.round(qs, 4))) > 5


def test_frame_rate_and_export(app):
    app.reset()
    t0 = time.perf_counter()
    n = 0
    from parttwin1d.geometry.cameras import look_at_c2w
    from parttwin1d.viewer.maniskill_viewer import sapien_pose_from_c2w

    c, r = app.object_center_radius()
    while time.perf_counter() - t0 < 2.0:  # orbit: Gaussian layer re-rendered every frame
        a = 0.05 * n
        d = np.array([np.cos(a), -np.sin(a) - 1.2, 0.8])
        app.viewer.set_camera_pose(sapien_pose_from_c2w(look_at_c2w(c + d / np.linalg.norm(d) * 3 * r, c)))
        frame(app, 1)
        n += 1
    fps = n / (time.perf_counter() - t0)
    (ART / "maniskill_viewer_perf.json").write_text(json.dumps({"orbit_fps": fps, **app.plugin.last_ms,
                                                                "resolution": [app.plugin.last_camera.width,
                                                                               app.plugin.last_camera.height]}))
    assert fps > 20, fps
    out = app.export()
    assert (out / "viewer_composited.png").exists() and (out / "front_oblique_meta.json").exists()
