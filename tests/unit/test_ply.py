"""Graphdeco PLY codec (CPU)."""
import numpy as np
import pytest

from parttwin1d.assets.gaussian_ply import (
    SH_C0, PLYFormatError, load_gaussian_ply, ordered_columns, pack_graphdeco_sh, unpack_graphdeco_sh,
    write_gaussian_ply,
)


def _write(path, n=5, degree=3, seed=0, **over):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2
    args = dict(means=rng.normal(size=(n, 3)), quats_wxyz=rng.normal(size=(n, 4)),
                scales=np.exp(rng.normal(-4, 0.3, size=(n, 3))), opacities=rng.uniform(0.1, 0.9, n),
                sh=rng.normal(size=(n, k, 3)))
    args.update(over)
    write_gaussian_ply(path, **args)
    return args


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_roundtrip_all_degrees(tmp_path, degree):
    a = _write(tmp_path / "g.ply", degree=degree)
    g = load_gaussian_ply(tmp_path / "g.ply")
    assert g.sh_degree == degree and g.sh.shape == (5, (degree + 1) ** 2, 3)
    np.testing.assert_allclose(g.means, a["means"], rtol=1e-6)
    np.testing.assert_allclose(g.sh, a["sh"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g.scales, a["scales"], rtol=1e-5)          # exp(log-scale)
    np.testing.assert_allclose(g.opacities, a["opacities"], rtol=1e-5)    # sigmoid(logit)
    q = a["quats_wxyz"] / np.linalg.norm(a["quats_wxyz"], axis=1, keepdims=True)
    np.testing.assert_allclose(g.quats, q, rtol=1e-5, atol=1e-6)


def test_channel_major_f_rest_layout():
    dc = np.array([[1, 2, 3]], np.float32)
    rest = np.array([[11, 12, 21, 22, 31, 32]], np.float32)  # R basis 1..2, G ..., B ...
    sh = unpack_graphdeco_sh(dc, rest)
    np.testing.assert_array_equal(sh[0, 1], [11, 21, 31])
    np.testing.assert_array_equal(sh[0, 2], [12, 22, 32])
    d2, r2 = pack_graphdeco_sh(sh)
    np.testing.assert_array_equal(r2, rest)


def test_deterministic_and_readonly(tmp_path):
    _write(tmp_path / "g.ply")
    a, b = load_gaussian_ply(tmp_path / "g.ply"), load_gaussian_ply(tmp_path / "g.ply")
    np.testing.assert_array_equal(a.sh, b.sh)
    with pytest.raises(ValueError):
        a.means[0, 0] = 1.0


def test_zero_quaternion_rejected(tmp_path):
    q = np.tile([1.0, 0, 0, 0], (5, 1))
    q[2] = 0
    _write(tmp_path / "g.ply", quats_wxyz=q)
    with pytest.raises(PLYFormatError, match="zero-norm"):
        load_gaussian_ply(tmp_path / "g.ply")


def test_missing_fields_and_point_cloud_rejected(tmp_path):
    from plyfile import PlyData, PlyElement

    arr = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(tmp_path / "pc.ply"))
    with pytest.raises(PLYFormatError, match="point cloud"):
        load_gaussian_ply(tmp_path / "pc.ply")


def test_gap_in_f_rest_rejected():
    with pytest.raises(PLYFormatError, match="continuously"):
        ordered_columns(("f_rest_0", "f_rest_2"), "f_rest_")


def test_invalid_basis_count_rejected():
    with pytest.raises(PLYFormatError):
        unpack_graphdeco_sh(np.zeros((1, 3)), np.zeros((1, 4)))


def test_dc_color_convention(tmp_path):
    sh = np.zeros((1, 1, 3))
    sh[0, 0] = (np.array([0.2, 0.5, 0.9]) - 0.5) / SH_C0
    _write(tmp_path / "g.ply", n=1, degree=0, sh=sh)
    g = load_gaussian_ply(tmp_path / "g.ply")
    np.testing.assert_allclose(g.sh[0, 0] * SH_C0 + 0.5, [0.2, 0.5, 0.9], atol=1e-6)


def test_fixture_plys_are_valid(root):
    b = load_gaussian_ply(root / "examples/hinged_box/base.ply")
    d = load_gaussian_ply(root / "examples/hinged_box/door.ply")
    assert b.sh_degree == 3 and d.sh_degree == 3
    assert b.count > 100 and d.count > 1000
