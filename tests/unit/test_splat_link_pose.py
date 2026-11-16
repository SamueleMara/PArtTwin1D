"""Per-link Gaussian transport on CPU tensors (no rasterizer): exact SE(3), no drift,
canonical-frame SH directions. The real CUDA render is covered in tests/integration."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from parttwin1d.config.schema import LinkBinding
from parttwin1d.geometry.se3 import axis_angle_matrix, quat_to_matrix_torch
from parttwin1d.rendering.gaussians import load_scene
from parttwin1d.rendering.link_adapter import LinkAdapter


@pytest.fixture()
def scene(hinged_cfg):
    return load_scene(hinged_cfg.links, device="cpu")


def poses(q, cfg):
    T_door = axis_angle_matrix([0, 0, 1], q)  # hinge at base origin, +z
    return torch.tensor(np.stack([np.eye(4), T_door]), dtype=torch.float32)


def test_static_link_invariant_and_moving_link_rotates(scene, hinged_cfg):
    ad = LinkAdapter(scene)
    m0 = ad.update(poses(0.0, hinged_cfg))[0].clone()
    m1 = ad.update(poses(1.0, hinged_cfg))[0].clone()
    base, door = scene.link_slice(0), scene.link_slice(1)
    torch.testing.assert_close(m0[base], m1[base], atol=0, rtol=0)  # static link bit-identical
    # door means rotate by exactly 1 rad about +z through the origin
    R = torch.tensor(Rotation.from_rotvec([0, 0, 1.0]).as_matrix(), dtype=torch.float32)
    torch.testing.assert_close(m1[door], m0[door] @ R.T, atol=2e-6, rtol=0)
    # z (hinge axis) coordinate unchanged, radius about hinge preserved
    torch.testing.assert_close(m1[door, 2], m0[door, 2], atol=1e-6, rtol=0)
    torch.testing.assert_close(m1[door, :2].norm(dim=1), m0[door, :2].norm(dim=1), atol=2e-6, rtol=0)


def test_orientation_composition_matches_matrix(scene, hinged_cfg):
    ad = LinkAdapter(scene)
    _, qw = ad.update(poses(0.7, hinged_cfg))
    door = scene.link_slice(1)
    T_wp = poses(0.7, hinged_cfg)[1] @ scene.T_link_ply[1]
    R_ref = T_wp[:3, :3] @ quat_to_matrix_torch(scene.quats[door])
    torch.testing.assert_close(quat_to_matrix_torch(qw[door]), R_ref, atol=1e-5, rtol=0)


def test_no_drift_and_canonical_immutable(scene, hinged_cfg):
    ad = LinkAdapter(scene)
    before = scene.checksum()
    m_ref = ad.update(poses(0.0, hinged_cfg))[0].clone()
    q_ref = ad.quats_w.clone()
    for q in np.linspace(0, 1.57, 200).tolist() + np.linspace(1.57, 0, 200).tolist():
        ad.update(poses(q, hinged_cfg))
    ad.update(poses(0.0, hinged_cfg))
    torch.testing.assert_close(ad.means_w, m_ref, atol=0, rtol=0)  # recomputed from canonical: exact
    torch.testing.assert_close(ad.quats_w, q_ref, atol=0, rtol=0)
    assert scene.checksum() == before


def test_sh_directions_are_canonical(scene, hinged_cfg, monkeypatch):
    """Camera rigidly co-moving with the door sees identical SH directions/colors."""
    import gsplat.cuda._wrapper as w

    captured = []

    def fake_sh(degree, dirs, coeffs):  # same shapes as gsplat 1.5.3: dirs [...,3], coeffs [...,K,3]
        captured.append(torch.nn.functional.normalize(dirs, dim=-1).clone())
        return torch.zeros(dirs.shape, dtype=dirs.dtype)

    monkeypatch.setattr(w, "spherical_harmonics", fake_sh)
    ad = LinkAdapter(scene)
    cam0 = np.array([1.2, -1.5, 1.05])
    ad.update(poses(0.0, hinged_cfg))
    ad.colors(torch.tensor(cam0[None], dtype=torch.float32), 3)
    T = axis_angle_matrix([0, 0, 1], 1.1)
    ad.update(poses(1.1, hinged_cfg))
    ad.colors(torch.tensor((T[:3, :3] @ cam0)[None], dtype=torch.float32), 3)
    door = scene.link_slice(1)
    torch.testing.assert_close(captured[0][0, door], captured[1][0, door], atol=1e-5, rtol=0)
    assert not torch.allclose(captured[0][0, scene.link_slice(0)], captured[1][0, scene.link_slice(0)], atol=1e-3)


def test_mixed_sh_degree_zero_padding(tmp_path):
    from parttwin1d.assets.gaussian_ply import write_gaussian_ply

    rng = np.random.default_rng(1)
    for name, deg in (("a", 0), ("b", 2)):
        write_gaussian_ply(tmp_path / f"{name}.ply", rng.normal(size=(4, 3)), np.tile([1.0, 0, 0, 0], (4, 1)),
                           np.full((4, 3), 0.01), np.full(4, 0.5), rng.normal(size=(4, (deg + 1) ** 2, 3)))
    sc = load_scene([LinkBinding("a", tmp_path / "a.ply", np.eye(4)), LinkBinding("b", tmp_path / "b.ply", np.eye(4))],
                    device="cpu")
    assert sc.sh.shape == (8, 9, 3) and sc.degrees == (0, 2)
    assert torch.count_nonzero(sc.sh[:4, 1:]) == 0
