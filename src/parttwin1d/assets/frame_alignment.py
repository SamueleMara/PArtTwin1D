"""Algebraic derivation and checking of T_link_ply — never visual guessing.

A reconstruction usually gives Gaussians of part i in a *capture world* frame
``Wc`` (e.g. COLMAP) at the reference articulation q0. With the explicitly
calibrated registration ``T_W_Wc`` (simulator world <- capture world) and the
PhysX link pose at the same q0, ``T_W_Li(q0)``::

    T_Li_Pi = T_W_Li(q0)^-1 @ T_W_Wc @ T_Wc_Pi

(``T_Wc_Pi`` is identity when the PLY is stored in capture-world coordinates.)
After this, ``T_W_Pi(q) = T_W_Li(q) @ T_Li_Pi`` places the part for every q.

Scale: ``T_W_Wc`` must be rigid. A reconstruction in arbitrary units has to be
rescaled *before* (write a metric PLY) and validated against collision geometry
with :func:`collision_containment` — scale is never hidden inside a 4x4.
"""
from __future__ import annotations

import numpy as np

from ..geometry.se3 import compose, invert, transform_points, validate_se3


def link_ply_from_capture(T_W_L_q0, T_W_Wc, T_Wc_P=None) -> np.ndarray:
    """Derive T_link_ply from the capture registration at reference q0."""
    T_Wc_P = np.eye(4) if T_Wc_P is None else T_Wc_P
    return compose(invert(T_W_L_q0), validate_se3(T_W_Wc, "T_W_Wc"), validate_se3(T_Wc_P, "T_Wc_P"))


def collision_containment(means_ply, T_link_ply, shapes, margin: float = 0.01) -> dict:
    """Fraction of Gaussian means inside the link's primitive collision shapes.

    A strong frame/scale error shows up as a low fraction. Mesh collisions are
    skipped (reported). Returns a dict suitable for the inspector/report.
    """
    pts_link = transform_points(T_link_ply, means_ply)
    inside = np.zeros(len(pts_link), dtype=bool)
    used, skipped = 0, 0
    for s in shapes:
        p = transform_points(invert(s.T_link_shape), pts_link)
        if s.kind == "box":
            half = np.asarray(s.size) / 2 + margin
            inside |= np.all(np.abs(p) <= half, axis=1)
        elif s.kind == "cylinder":
            r, length = s.size
            inside |= (np.linalg.norm(p[:, :2], axis=1) <= r + margin) & (np.abs(p[:, 2]) <= length / 2 + margin)
        elif s.kind == "sphere":
            inside |= np.linalg.norm(p, axis=1) <= s.size[0] + margin
        else:
            skipped += 1
            continue
        used += 1
    frac = float(inside.mean()) if used else float("nan")
    return {"fraction_inside": frac, "shapes_used": used, "shapes_skipped": skipped,
            "margin_m": margin, "count": int(len(pts_link))}
