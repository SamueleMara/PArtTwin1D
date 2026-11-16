#!/usr/bin/env python3
"""Regenerate the deterministic hinged-box Gaussian fixture (synthetic).

NOT a reconstruction and NOT a substitute for real object PLYs. Designed to
exercise the frame contract:

* ``base.ply`` is stored directly in ``base_link`` coordinates
  (``T_link_ply = I`` stated explicitly in the config) and contains an
  RGB axis triad (+X red, +Y green, +Z blue) at (-0.15, 0, 0) for viewer
  convention checks.
* ``door.ply`` is stored in a *different* frame ``P_door``: centred on the
  panel and rotated +90 deg about X. ``T_link_ply`` (``DOOR_T_LINK_PLY``)
  maps it into ``door_link``; dropping it would visibly misplace the door.
* Both PLYs carry degree-3 SH (deterministic small view-dependent terms).

Usage:  python examples/hinged_box/generate_fixture.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from parttwin1d.assets.gaussian_ply import SH_C0, write_gaussian_ply  # noqa: E402

DOOR_T_LINK_PLY = np.eye(4)
DOOR_T_LINK_PLY[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
DOOR_T_LINK_PLY[:3, 3] = [0.265, 0.0, 0.34]


def _sh(rgb, n, rng, view_amp):
    sh = np.zeros((n, 16, 3), np.float32)
    sh[:, 0, :] = (np.asarray(rgb, np.float32) - 0.5) / SH_C0
    sh[:, 1:, :] = rng.normal(0.0, view_amp, size=(n, 15, 3)).astype(np.float32)
    return sh


def _segment(a, b, step):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def base_gaussians(rng):
    pts, cols = [], []
    gray = (0.55, 0.57, 0.62)
    for seg in (((0, -0.03, -0.01), (0, -0.03, 0.69)), ((0.52, -0.03, -0.01), (0.52, -0.03, 0.69)),
                ((0, -0.03, 0.0), (0.52, -0.03, 0.0)), ((0, -0.03, 0.68), (0.52, -0.03, 0.68))):
        p = _segment(*seg, step=0.008)
        pts.append(p)
        cols.append(np.tile(gray, (len(p), 1)))
    o = np.array([-0.15, 0.0, 0.0])
    for axis, rgb in ((0, (0.9, 0.1, 0.1)), (1, (0.1, 0.8, 0.1)), (2, (0.1, 0.2, 0.95))):
        e = np.zeros(3)
        e[axis] = 0.15
        p = _segment(o + 0.01 * e / 0.15, o + e, step=0.006)
        pts.append(p)
        cols.append(np.tile(rgb, (len(p), 1)))
    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    n = len(pts)
    quats = np.tile([1.0, 0, 0, 0], (n, 1))
    scales = np.tile([0.008, 0.008, 0.008], (n, 1))
    sh = _sh((0, 0, 0), n, rng, 0.01)
    sh[:, 0, :] = ((cols - 0.5) / SH_C0).astype(np.float32)
    return pts, quats, scales, np.full(n, 0.98), sh


def door_gaussians(rng):
    xs = np.arange(0.035, 0.5, 0.014)
    zs = np.arange(0.03, 0.655, 0.014)
    xx, zz = np.meshgrid(xs, zs)
    p_link = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], -1)
    n = len(p_link)
    # Two-tone panel (red with darker handle strip) so rotation is visible.
    rgb = np.tile([0.82, 0.2, 0.16], (n, 1))
    rgb[(p_link[:, 0] > 0.42) & (np.abs(p_link[:, 2] - 0.34) < 0.06)] = [0.15, 0.12, 0.1]
    # Express in P_door: x_P = T_P_L x_L.
    R_LP, t_LP = DOOR_T_LINK_PLY[:3, :3], DOOR_T_LINK_PLY[:3, 3]
    p_ply = (p_link - t_LP) @ R_LP  # = R_LP^T (x_L - t)
    # Flat Gaussians in the link frame (thin along link y = panel normal);
    # their orientation in P is R_PL = R_LP^T.
    q_pl = Rotation.from_matrix(R_LP.T).as_quat()[[3, 0, 1, 2]]
    quats = np.tile(q_pl, (n, 1))
    scales = np.tile([0.009, 0.0025, 0.009], (n, 1))
    sh = _sh((0, 0, 0), n, rng, 0.04)
    sh[:, 0, :] = ((rgb - 0.5) / SH_C0).astype(np.float32)
    return p_ply, quats, scales, np.full(n, 0.985), sh


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(20261116)
    write_gaussian_ply(args.out / "base.ply", *base_gaussians(rng))
    write_gaussian_ply(args.out / "door.ply", *door_gaussians(rng))
    print(f"wrote {args.out / 'base.ply'} and {args.out / 'door.ply'}")
    print("door T_link_ply =", np.array2string(DOOR_T_LINK_PLY, precision=6, separator=", "))


if __name__ == "__main__":
    main()
