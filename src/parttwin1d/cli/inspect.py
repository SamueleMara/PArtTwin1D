"""Field-by-field manifest inspector (no SAPIEN, no CUDA).

    python -m parttwin1d.cli.inspect --config configs/examples/hinged_box.yaml
Exit code 0 when no errors were found.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def inspect_config(path) -> "object":
    from ..assets.frame_alignment import collision_containment
    from ..assets.gaussian_ply import load_gaussian_ply
    from ..assets.urdf import parse_urdf
    from ..config.loader import load_config
    from ..config.validate import ConfigError, Report
    from ..geometry.cameras import project_points

    rep = Report()
    try:
        cfg = load_config(path, check_files=False)
        rep.ok("config", f"{cfg.source} (schema v{cfg.schema_version}, object {cfg.object_name!r})")
    except ConfigError as exc:
        rep.error("config", str(exc))
        return rep
    try:
        urdf = parse_urdf(cfg.urdf)
        rep.ok("urdf", f"{cfg.urdf} — {len(urdf.links)} links, {len(urdf.joints)} joints, root {urdf.root_links[0]!r}")
    except Exception as exc:
        rep.error("urdf", str(exc))
        return rep
    for j in urdf.joints.values():
        lim = f"[{j.lower}, {j.upper}] {j.unit}" if j.lower is not None else "unbounded"
        kind = "movable" if j.movable else ("mimic" if j.mimic else "fixed/other")
        rep.ok(f"urdf.joint.{j.name}", f"{j.type} ({kind}) {j.parent} -> {j.child}, axis {np.round(j.axis, 4).tolist()}, "
                                       f"limits {lim}, damping {j.damping}, friction {j.friction}")
        if j.type == "continuous":
            rep.warn(f"urdf.joint.{j.name}", "continuous joint: GUI slider shows ±π, sweep disabled")
    for name, link in urdf.links.items():
        prim = [s.kind for s in link.collisions]
        if not link.collisions:
            rep.warn(f"urdf.link.{name}", "no collision geometry (contacts impossible in dynamics mode)")
        if link.mass is None:
            rep.warn(f"urdf.link.{name}", "no <inertial> mass (PhysX will infer one)")
        else:
            rep.ok(f"urdf.link.{name}", f"mass {link.mass} kg, collisions {prim}")
    rep.ok("T_world_root", f"proper rigid transform, t = {np.round(cfg.T_world_root[:3, 3], 6).tolist()}")
    bound = set()
    for b in cfg.links:
        f = f"links.{b.link}"
        if b.link not in urdf.links:
            rep.error(f, f"not a URDF link; URDF links: {sorted(urdf.links)}")
            continue
        bound.add(b.link)
        if not b.ply.is_file():
            rep.error(f"{f}.ply", f"missing file {b.ply}")
            continue
        try:
            g = load_gaussian_ply(b.ply)
        except Exception as exc:
            rep.error(f"{f}.ply", str(exc))
            continue
        lo, hi = g.aabb()
        rep.ok(f"{f}.ply", f"{b.ply.name}: {g.count} Gaussians, SH degree {g.sh_degree}, "
                           f"AABB(PLY) {np.round(lo, 4).tolist()}..{np.round(hi, 4).tolist()} m")
        ident = np.allclose(b.T_link_ply, np.eye(4))
        rep.ok(f"{f}.T_link_ply", "explicit identity" if ident else
               f"rigid, t = {np.round(b.T_link_ply[:3, 3], 6).tolist()}")
        ext = float(np.max(hi - lo))
        if ext > 20 or ext < 1e-3:
            rep.warn(f"{f}.ply", f"extent {ext:.4g} m is implausible for metres — check scale")
        cc = collision_containment(g.means, b.T_link_ply, urdf.links[b.link].collisions)
        if cc["shapes_used"]:
            lvl = rep.ok if cc["fraction_inside"] >= 0.9 else rep.warn
            lvl(f"{f}.alignment", f"{100 * cc['fraction_inside']:.1f}% of Gaussian means inside the link's "
                                  f"collision primitives (margin {cc['margin_m']} m)")
        else:
            rep.warn(f"{f}.alignment", "no primitive collision shapes to check alignment against")
    unbound = [n for n in urdf.links if n not in bound]
    if unbound:
        rep.warn("links", f"URDF links without Gaussians (invisible): {unbound}")
    for jn, q in cfg.initial_q.items():
        j = urdf.joints.get(jn)
        if j is None or not j.movable:
            rep.error(f"initial_q.{jn}", "not a movable URDF joint")
        elif j.lower is not None and not (j.lower <= q <= j.upper):
            rep.error(f"initial_q.{jn}", f"{q} outside [{j.lower}, {j.upper}]")
        else:
            rep.ok(f"initial_q.{jn}", f"{q} {j.unit}")
    for c in cfg.cameras:
        K = c.K
        f = f"cameras.{c.name}"
        rep.ok(f, f"{c.width}x{c.height} fx={K[0,0]:g} fy={K[1,1]:g} cx={K[0,2]:g} cy={K[1,2]:g} (skew 0); "
                  f"position {np.round(c.T_world_camera[:3, 3], 4).tolist()} m")
        uv, z = project_points(K, c.T_world_camera, cfg.T_world_root[:3, 3][None])
        if z[0] <= 0:
            rep.warn(f, "object root is behind this camera (+Z optical axis points away)")
        elif not (0 <= uv[0, 0] < c.width and 0 <= uv[0, 1] < c.height):
            rep.warn(f, f"object root projects outside the image at {np.round(uv[0], 1).tolist()}")
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="parttwin inspect", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    rep = inspect_config(args.config)
    print(rep.format())
    n_err, n_warn = len(rep.errors), sum(d.level == "warn" for d in rep.items)
    print(f"\n{'FAILED' if n_err else 'OK'}: {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
