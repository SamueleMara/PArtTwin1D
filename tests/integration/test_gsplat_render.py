"""REAL gsplat 1.5.3 CUDA rasterization tests (no mocks, no fallback renderer).

Opt-in:  PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/integration
Poses come from real ManiSkill/SAPIEN (physx_cpu) unless a test is explicitly
about the renderer's frame algebra, where explicit link poses are fed in.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import ROOT, requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu]
ART = ROOT / "outputs" / "test_artifacts"


@pytest.fixture(scope="module")
def stack(hinged_cfg):
    import torch

    from parttwin1d.rendering.gsplat_backend import ArticulatedGaussianRenderer
    from parttwin1d.sim.articulation import ArticulationDriver
    from parttwin1d.sim.env import create_env

    assert torch.cuda.is_available()
    env = create_env(hinged_cfg, "physx_cpu")
    env.reset(seed=0)
    drv = ArticulationDriver(env, hinged_cfg)
    r = ArticulatedGaussianRenderer.from_config(hinged_cfg)
    r.bind_articulation(env.twin)
    yield hinged_cfg, drv, r
    env.close()


def cam0(cfg):
    from parttwin1d.rendering.cameras import RenderCamera

    return RenderCamera.from_spec(cfg.cameras[0])


def test_real_kernel_no_fallback(stack):
    _, _, r = stack
    from gsplat.cuda._backend import _C

    assert _C is not None and r.extension_path.endswith("gsplat_cuda.so")
    assert "torch_extensions" in r.extension_path


def test_render_changes_with_joint_and_static_link_fixed(stack):
    import torch

    cfg, drv, r = stack
    cam = cam0(cfg)
    before = r.scene.checksum()
    out = {}
    for q in (0.0, 1.1, 0.0):
        drv.set_joint_kinematic("door_hinge", q)
        s = drv.snapshot("kinematic")
        r.update_from_snapshot(s)
        res = r.render(cam, segmentation=True).validate()
        out.setdefault(q, []).append((res, r.world_means().clone()))
    (a, ma), (c, mc) = out[0.0]
    b, mb = out[1.1][0]
    base, door = r.scene.link_slice(0), r.scene.link_slice(1)
    assert a.rgb.device.type == "cuda" and a.rgb.shape == (480, 640, 3)
    assert int(a.valid.sum()) > 5000 and int(b.valid.sum()) > 5000
    assert torch.isfinite(a.rgb).all()
    d = a.depth[a.valid]
    assert float(d.min()) > 1.0 and float(d.max()) < 3.0       # camera is ~1.6–2.4 m from the object
    assert (a.rgb - b.rgb).abs().mean().item() > 0.01          # image changes with q
    torch.testing.assert_close(ma[base], mb[base], atol=0, rtol=0)  # static base: identical world Gaussians
    assert (ma[door] - mb[door]).norm(dim=1).max().item() > 0.3
    torch.testing.assert_close(a.rgb, c.rgb, atol=0, rtol=0)   # back at q0: bit-identical (no drift)
    assert r.scene.checksum() == before                        # canonical buffers immutable
    assert set(torch.unique(a.link_id).tolist()) == {-1, 0, 1}


def _single_link_renderer(cfg, link="door_link"):
    from parttwin1d.rendering.gaussians import load_scene
    from parttwin1d.rendering.gsplat_backend import ArticulatedGaussianRenderer

    return ArticulatedGaussianRenderer(load_scene([cfg.link_binding(link)]), background=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_sh_rigid_invariance(hinged_cfg, degree):
    """Rigidly moving link + rigidly co-moving camera => identical image for every SH degree."""
    from parttwin1d.geometry.se3 import axis_angle_matrix
    from parttwin1d.rendering.cameras import RenderCamera

    r = _single_link_renderer(hinged_cfg)
    c0 = cam0(hinged_cfg)
    G = axis_angle_matrix([0.3, -0.5, 0.8], 1.2, pivot=[0.1, 0.2, 0.3])
    G[:3, 3] += [0.2, -0.1, 0.05]
    r.update_from_link_poses({"door_link": np.eye(4)})
    a = r.render(c0, sh_degree=degree)
    r.update_from_link_poses({"door_link": G})
    b = r.render(RenderCamera(c0.name, c0.width, c0.height, c0.K, G @ c0.T_world_camera), sh_degree=degree)
    err = (a.rgb - b.rgb).abs().max().item()
    assert int(a.valid.sum()) > 5000
    assert err < 2e-3, f"degree {degree}: max |ΔRGB| {err}"


def test_world_frame_sh_would_be_wrong(hinged_cfg):
    """Negative control: gsplat's built-in SH (world-frame directions) is NOT rigid-invariant
    for a rotated link — this is why SH is evaluated in the canonical PLY frame."""
    import torch
    from gsplat.rendering import rasterization

    from parttwin1d.geometry.se3 import axis_angle_matrix

    r = _single_link_renderer(hinged_cfg)
    c0 = cam0(hinged_cfg)
    G = axis_angle_matrix([0, 0, 1], 1.2)

    def builtin(T_link, c2w):
        r.update_from_link_poses({"door_link": T_link})
        vm = torch.tensor(np.linalg.inv(c2w)[None], dtype=torch.float32, device="cuda")
        K = torch.tensor(c0.K[None], dtype=torch.float32, device="cuda")
        out, _, _ = rasterization(r.adapter.means_w, r.adapter.quats_w, r.scene.scales, r.scene.opacities,
                                  r.scene.sh, vm, K, c0.width, c0.height, sh_degree=3, packed=False)
        return out[0]
    diff = (builtin(np.eye(4), c0.T_world_camera) - builtin(G, G @ c0.T_world_camera)).abs().max().item()
    assert diff > 2e-2, diff


def test_canonical_sh_equals_gsplat_reference_at_identity(hinged_cfg):
    """With T_world_ply = I our canonical-frame SH must reproduce gsplat's own SH path."""
    import torch
    from gsplat.rendering import rasterization

    r = _single_link_renderer(hinged_cfg)
    c0 = cam0(hinged_cfg)
    T_lp = hinged_cfg.link_binding("door_link").T_link_ply
    r.update_from_link_poses({"door_link": np.linalg.inv(T_lp)})  # => T_world_ply = I
    ours = r.render(c0, sh_degree=3)
    vm = torch.tensor(c0.viewmat[None], dtype=torch.float32, device="cuda")
    K = torch.tensor(c0.K[None], dtype=torch.float32, device="cuda")
    ref, _, _ = rasterization(r.adapter.means_w, r.adapter.quats_w, r.scene.scales, r.scene.opacities, r.scene.sh,
                              vm, K, c0.width, c0.height, sh_degree=3, packed=False,
                              backgrounds=torch.zeros((1, 3), device="cuda"))
    assert (ours.rgb - ref[0].clamp(0, 1)).abs().max().item() < 2e-3


def test_depth_occlusion_independent_of_link_order(tmp_path, root):
    """Two overlapping coloured planes on different links: the nearer one wins, whatever the order."""
    import torch

    from parttwin1d.assets.gaussian_ply import SH_C0, write_gaussian_ply
    from parttwin1d.config.schema import LinkBinding
    from parttwin1d.geometry.cameras import look_at_c2w
    from parttwin1d.rendering.cameras import RenderCamera
    from parttwin1d.rendering.gaussians import load_scene
    from parttwin1d.rendering.gsplat_backend import ArticulatedGaussianRenderer

    xs, zs = np.meshgrid(np.arange(-0.2, 0.2, 0.01), np.arange(-0.2, 0.2, 0.01))
    plane = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], -1)
    n = len(plane)
    for name, rgb in (("red", (0.9, 0.1, 0.1)), ("blue", (0.1, 0.1, 0.9))):
        sh = np.zeros((n, 1, 3))
        sh[:, 0] = (np.array(rgb) - 0.5) / SH_C0
        write_gaussian_ply(tmp_path / f"{name}.ply", plane, np.tile([1.0, 0, 0, 0], (n, 1)),
                           np.tile([0.008, 0.002, 0.008], (n, 1)), np.full(n, 0.99), sh)
    cam = RenderCamera("c", 320, 240, [[300.0, 0, 160], [0, 300, 120], [0, 0, 1]], look_at_c2w([0, -1.0, 0], [0, 0, 0]))
    b_red = LinkBinding("red", tmp_path / "red.ply", np.eye(4))
    b_blue = LinkBinding("blue", tmp_path / "blue.ply", np.eye(4))

    def render(order, red_y, blue_y):
        r = ArticulatedGaussianRenderer(load_scene(order), background=(1, 1, 1))
        Tr, Tb = np.eye(4), np.eye(4)
        Tr[1, 3], Tb[1, 3] = red_y, blue_y
        r.update_from_link_poses({"red": Tr, "blue": Tb})
        return r.render(cam).rgb[100:140, 140:180].mean((0, 1))

    for red_y, blue_y, winner in ((0.0, 0.1, 0), (0.1, 0.0, 2)):  # camera at y=-1: smaller y is nearer
        c1 = render([b_red, b_blue], red_y, blue_y)
        c2 = render([b_blue, b_red], red_y, blue_y)
        torch.testing.assert_close(c1, c2, atol=1e-4, rtol=0)
        assert c1[winner] > 0.8 and c1[2 - winner] < 0.2, c1


def test_exact_principal_point_and_focal(stack):
    """Shifting cx/cy by integer pixels shifts the image exactly; fx != fy is honoured."""
    from parttwin1d.rendering.cameras import RenderCamera

    cfg, drv, r = stack
    drv.set_joint_kinematic("door_hinge", 0.4)
    r.update_from_snapshot(drv.snapshot("kinematic"))
    c = cam0(cfg)
    K2 = c.K.copy()
    K2[0, 2] += 12
    K2[1, 2] -= 7
    a = r.render(c).alpha.cpu().numpy()
    b = r.render(RenderCamera(c.name, c.width, c.height, K2, c.T_world_camera)).alpha.cpu().numpy()
    # u' = u + 12, v' = v - 7  =>  b[v, u] = a[v + 7, u - 12]
    H, W = a.shape
    m = 4
    np.testing.assert_allclose(b[m:H - 7 - m, 12 + m:W - m], a[7 + m:H - m, m:W - 12 - m], atol=2e-3)
    assert a.max() > 0.5
    K3 = c.K.copy()
    K3[1, 1] *= 1.5  # stretch vertically only
    v = r.render(RenderCamera(c.name, c.width, c.height, K3, c.T_world_camera)).alpha.cpu().numpy()
    rows = lambda m: np.ptp(np.nonzero(m.max(1) > 0.5)[0])
    cols = lambda m: np.ptp(np.nonzero(m.max(0) > 0.5)[0])
    assert rows(v) > 1.3 * rows(a) and abs(cols(v) - cols(a)) <= 2


def test_export_roundtrip(stack, tmp_path):
    from parttwin1d.rendering.observations import export_result, load_export, rgb_uint8

    cfg, drv, r = stack
    drv.set_joint_kinematic("door_hinge", 0.8)
    s = drv.snapshot("kinematic")
    r.update_from_snapshot(s)
    res = r.render(cam0(cfg), segmentation=True)
    paths = export_result(res, tmp_path, "x")
    back = load_export(paths["meta"])
    np.testing.assert_array_equal(back["rgb_png"], rgb_uint8(res.rgb.cpu().numpy()))
    np.testing.assert_array_equal(back["depth"], res.depth.cpu().numpy())
    np.testing.assert_array_equal(back["link_id"], res.link_id.cpu().numpy())
    md = back["meta"]
    np.testing.assert_allclose(md["camera"]["K"], cfg.cameras[0].K)
    np.testing.assert_allclose(md["camera"]["T_world_camera_c2w_opencv"], cfg.cameras[0].T_world_camera)
    assert md["q"] == {"door_hinge": pytest.approx(0.8, abs=1e-6)} and md["revision"] == s.revision
    np.testing.assert_allclose(md["T_world_link"]["door_link"], s.T_world_link["door_link"])


def test_gui_callback_equals_offline_render(stack):
    """The nerfview render_fn path and the offline path agree for the same q/K/c2w (pre-JPEG)."""
    import nerfview

    from parttwin1d.rendering.cameras import RenderCamera
    from parttwin1d.rendering.observations import rgb_uint8
    from parttwin1d.viewer.render_callback import ViewerRenderCallback

    cfg, drv, r = stack
    drv.set_joint_kinematic("door_hinge", 0.6)
    snap = drv.snapshot("kinematic")

    class Snap:
        def snapshot(self):
            return snap
    cb = ViewerRenderCallback(r, Snap())
    c2w = cfg.cameras[0].T_world_camera
    cs = nerfview.CameraState(fov=0.8, aspect=640 / 480, c2w=c2w)
    tab = nerfview.RenderTabState()
    tab.viewer_width, tab.viewer_height = 640, 480
    img, depth = cb(cs, tab)
    r.update_from_snapshot(snap)
    off = r.render(RenderCamera.virtual(c2w, 0.8, 640, 480))
    np.testing.assert_array_equal(img, rgb_uint8(off.rgb.cpu().numpy()))
    assert img.dtype == np.uint8 and img.shape == (480, 640, 3) and depth.dtype == np.float32
    assert cb.count == 1 and cb.latest().revision == snap.revision
    # errors are rendered + reported, never raised into nerfview (which would os._exit)
    errs = []
    bad = ViewerRenderCallback(r, Snap(), on_error=lambda *a: errs.append(a))
    tab.viewer_width = 0
    out = bad(cs, tab)
    assert errs and out.dtype == np.uint8


def test_profile_steady_state(stack):
    from parttwin1d.evaluation.profiling import profile_render
    from parttwin1d.rendering.cameras import RenderCamera

    cfg, drv, r = stack
    r.update_from_snapshot(drv.snapshot("kinematic"))
    c = cam0(cfg)
    prof = {"640x480": profile_render(r, c), "640x480+linkid": profile_render(r, c, segmentation=True),
            "1280x960": profile_render(r, c.resized(1280, 960))}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "render_profile.json").write_text(json.dumps(prof, indent=2))
    assert prof["640x480"]["steady_fps"] > 30
    assert prof["640x480"]["peak_gpu_mem_mb"] < 1024
