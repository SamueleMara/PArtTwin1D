"""Config schema / URDF binding validation (CPU, no SAPIEN)."""
import os

import numpy as np
import pytest

from parttwin1d.config import ConfigError, load_config
from tests.conftest import base_config_dict, write_config


def test_example_config_loads(hinged_cfg):
    c = hinged_cfg
    assert c.schema_version == 1 and c.object_name == "hinged_box"
    assert c.link_names == ("base_link", "door_link")
    K = c.cameras[0].K
    assert (K[0, 0], K[1, 1], K[0, 2], K[1, 2]) == (600.0, 590.0, 331.5, 236.0)  # exact, not fx=fy/centred
    np.testing.assert_allclose(c.link_binding("door_link").T_link_ply[:3, 3], [0.265, 0, 0.34])
    assert c.urdf.is_absolute() and all(b.ply.is_absolute() for b in c.links)


def test_paths_independent_of_cwd(hinged_cfg_path, tmp_path):
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        c = load_config(hinged_cfg_path)
        assert c.links[0].ply.is_file()
    finally:
        os.chdir(old)


def _mut(root, tmp_path, fn, name="c.yaml"):
    d = base_config_dict(root)
    fn(d)
    return write_config(tmp_path / name, d)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d["links"].__setitem__("nonexistent_link", dict(d["links"]["door_link"], ply=d["links"]["door_link"]["ply"].replace("door", "base"))), "same file|not in URDF"),
    (lambda d: d["links"]["door_link"].pop("T_link_ply"), "T_link_ply: required"),
    (lambda d: d.pop("T_world_root"), "T_world_root: required"),
    (lambda d: d["links"]["door_link"].__setitem__("ply", "/nonexistent/door.ply"), "file not found"),
    (lambda d: d["links"]["door_link"].__setitem__("T_link_ply", [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), "orthonormal"),
    (lambda d: d["links"]["door_link"].__setitem__("T_link_ply", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "4x4"),
    (lambda d: d.__setitem__("unknown_key", 1), "unknown key"),
    (lambda d: d["links"]["door_link"].__setitem__("T_link_ply ", np.eye(4).tolist()), "unknown key"),
    (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
    (lambda d: d["cameras"][0].__setitem__("K", [[600, 2, 320], [0, 600, 240], [0, 0, 1]]), "skew"),
    (lambda d: d["cameras"][0].__setitem__("distortion", [0.1, 0, 0, 0, 0]), "distortion"),
    (lambda d: d["cameras"].append(dict(d["cameras"][0])), "duplicate"),
    (lambda d: d.__setitem__("initial_q", {"door_hinge": 3.0}), "outside URDF limits"),
    (lambda d: d.__setitem__("initial_q", {"base_link": 0.0}), "not a movable"),
])
def test_invalid_configs_rejected(root, tmp_path, mutate, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(_mut(root, tmp_path, mutate))


def test_duplicate_ply_binding_rejected(root, tmp_path):
    def m(d):
        d["links"]["door_link"]["ply"] = d["links"]["base_link"]["ply"]
    with pytest.raises(ConfigError, match="already bound"):
        load_config(_mut(root, tmp_path, m))


def test_json_supported(root, tmp_path):
    import json

    d = base_config_dict(root)
    (tmp_path / "c.json").write_text(json.dumps(d))
    assert load_config(tmp_path / "c.json").object_name == "hinged_box"


def test_urdf_parse_and_bindings(hinged_cfg):
    from parttwin1d.assets.urdf import parse_urdf

    m = parse_urdf(hinged_cfg.urdf)
    j = m.joints["door_hinge"]
    assert j.movable and j.unit == "rad" and (j.lower, j.upper) == (0.0, 1.57)
    np.testing.assert_allclose(j.axis, [0, 0, 1])
    assert m.root_links == ["base_link"]
    assert len(m.links["base_link"].collisions) == 5


def test_fixture_alignment_by_collision_containment(hinged_cfg):
    """T_link_ply is checked algebraically against URDF collision geometry, not by eye."""
    from parttwin1d.assets.frame_alignment import collision_containment
    from parttwin1d.assets.gaussian_ply import load_gaussian_ply
    from parttwin1d.assets.urdf import parse_urdf

    m = parse_urdf(hinged_cfg.urdf)
    for b in hinged_cfg.links:
        g = load_gaussian_ply(b.ply)
        cc = collision_containment(g.means, b.T_link_ply, m.links[b.link].collisions)
        assert cc["fraction_inside"] > 0.99, (b.link, cc)
    # dropping the door's T_link_ply is detected
    door = hinged_cfg.link_binding("door_link")
    cc = collision_containment(load_gaussian_ply(door.ply).means, np.eye(4), m.links["door_link"].collisions)
    assert cc["fraction_inside"] < 0.5


def test_link_ply_from_capture_algebra():
    from parttwin1d.assets.frame_alignment import link_ply_from_capture
    from parttwin1d.geometry.se3 import axis_angle_matrix, transform_points

    T_W_L = axis_angle_matrix([0, 0, 1], 0.4, pivot=[0.1, 0.2, 0])
    T_W_Wc = axis_angle_matrix([1, 0, 0], -0.3)
    T_W_Wc[:3, 3] = [0.5, -0.2, 1.0]
    T_L_P = link_ply_from_capture(T_W_L, T_W_Wc)
    x = np.array([[0.3, 0.1, -0.2]])
    np.testing.assert_allclose(transform_points(T_W_L @ T_L_P, x), transform_points(T_W_Wc, x), atol=1e-12)
