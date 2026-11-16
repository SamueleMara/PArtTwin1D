"""Offline kinematic joint sweep: real ManiSkill link poses -> gsplat -> files.

    python -m parttwin1d.cli.render --config configs/examples/hinged_box.yaml \
        --joint door_hinge --q-start 0 --q-end 1.3 --frames 20 --out outputs/hinged_box

Writes per camera and frame: RGB PNG, alpha PNG, depth .npy, arrays .npz,
metadata JSON (K, c2w, w2c, q, T_world_link), plus a GIF per camera and
``report.json`` with geometric consistency checks and render timing
(compile/first-frame vs steady state). This is a kinematic pose sweep, not a
physics trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(prog="parttwin render", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--joint", help="active joint to sweep (optional for 1-DoF objects)")
    p.add_argument("--q-start", type=float, default=None)
    p.add_argument("--q-end", type=float, default=None)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("outputs/render"))
    p.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default=None)
    p.add_argument("--camera", action="append", help="calibrated camera name(s); default all")
    p.add_argument("--segmentation", action="store_true", help="also export the link-ID pass")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 2:
        raise SystemExit("--frames must be >= 2")
    from ..config import load_config
    from ..evaluation.consistency import expected_child_pose, pose_error, static_link_invariance
    from ..evaluation.profiling import profile_render
    from ..rendering.cameras import RenderCamera
    from ..rendering.gsplat_backend import ArticulatedGaussianRenderer
    from ..rendering.observations import export_result, rgb_uint8, save_gif
    from ..sim.articulation import ArticulationDriver
    from ..sim.env import create_env

    cfg = load_config(args.config)
    cams = [RenderCamera.from_spec(c) for c in cfg.cameras if not args.camera or c.name in args.camera]
    if not cams:
        raise SystemExit("no calibrated cameras selected (config 'cameras' / --camera)")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    env = create_env(cfg, args.sim_backend)
    try:
        env.reset(seed=0)
        drv = ArticulationDriver(env, cfg)
        names = [j.name for j in drv.joints]
        if not names:
            raise SystemExit("URDF has no movable joints")
        jname = args.joint or (names[0] if len(names) == 1 else None)
        if jname is None or jname not in names:
            raise SystemExit(f"--joint required; active joints: {names}")
        j = drv.joints[drv.joint_index[jname]]
        qs = j.lower if args.q_start is None else args.q_start
        qe = j.upper if args.q_end is None else args.q_end
        if not np.isfinite([qs, qe]).all():
            raise SystemExit(f"{jname}: finite --q-start/--q-end required (unbounded joint)")
        print(f"[render] {cfg.urdf.name}: links {sorted(env.twin.links_map)}, active joints {names}", flush=True)
        t_sim = time.perf_counter() - t0
        t1 = time.perf_counter()
        renderer = ArticulatedGaussianRenderer.from_config(cfg)
        renderer.bind_articulation(env.twin)
        print(f"[render] loaded {renderer.gaussian_count} Gaussians across {len(cfg.links)} links "
              f"(SH degree {renderer.sh_degree}); gsplat ext {renderer.extension_path}", flush=True)
        t_load = time.perf_counter() - t1

        snaps, frames = [], {c.name: [] for c in cams}
        q_values = np.linspace(qs, qe, args.frames)
        drv.set_joint_kinematic(jname, float(q_values[0]))
        s0 = drv.snapshot("kinematic")
        renderer.update_from_snapshot(s0)
        t2 = time.perf_counter()
        renderer.render(cams[0])  # first real kernel launch (lazy CUDA init / allocator)
        first_ms = (time.perf_counter() - t2) * 1e3
        print(f"[render] first rasterization {first_ms:.1f} ms", flush=True)
        for i, q in enumerate(q_values):
            drv.set_joint_kinematic(jname, float(q))
            s = drv.snapshot("kinematic")
            snaps.append(s)
            renderer.update_from_snapshot(s)
            for cam in cams:
                res = renderer.render(cam, segmentation=args.segmentation).validate()
                if int(res.valid.sum()) == 0:
                    print(f"WARNING: frame {i} camera {cam.name}: no visible alpha", flush=True)
                export_result(res, args.out / cam.name, f"{i:04d}", {"camera_kind": "calibrated exact K"})
                frames[cam.name].append(rgb_uint8(res.rgb.detach().cpu().numpy()))
            print(f"[render] frame {i + 1}/{len(q_values)} {jname}={q:.5f} {j.unit}", flush=True)

        # Geometric checks against the analytic joint motion (single joint sweep).
        uj = drv.urdf.joints[jname]
        T0 = snaps[0].T_world_link[uj.child] if uj.child in snaps[0].T_world_link else None
        checks = {}
        if T0 is not None:
            errs = [pose_error(expected_child_pose(T0, uj, float(s.q[j.index]), float(snaps[0].q[j.index])),
                               s.T_world_link[uj.child]) for s in snaps]
            checks["moving_link_max_translation_err_m"] = max(e[0] for e in errs)
            checks["moving_link_max_rotation_err_rad"] = max(e[1] for e in errs)
        static = [b.link for b in cfg.links if b.link != uj.child and drv.urdf.joint_for_child(b.link) is None]
        for link in static:
            tr, rot = static_link_invariance(snaps, link)
            checks[f"static_{link}_max_translation_m"] = tr
            checks[f"static_{link}_max_rotation_rad"] = rot
        report = {"config": str(cfg.source), "urdf": str(cfg.urdf), "joint": jname, "unit": j.unit,
                  "q_start": qs, "q_end": qe, "frames": args.frames, "sim_backend": args.sim_backend or cfg.sim.backend,
                  "gaussians": renderer.gaussian_count, "sh_degree": renderer.sh_degree,
                  "timing_s": {"scene_setup": t_sim, "ply_load_upload": t_load}, "first_render_ms": first_ms,
                  "checks": checks, "cameras": {}, "sequence": [s.to_json() for s in snaps]}
        for cam in cams:
            fr = frames[cam.name]
            save_gif(fr, args.out / f"{cam.name}.gif", 120)
            mad = float(np.abs(fr[0].astype(np.float32) - fr[-1].astype(np.float32)).mean())
            report["cameras"][cam.name] = {"first_last_mean_abs_rgb_change_0_255": mad,
                                          "profile": profile_render(renderer, cam, n_warmup=5, n_frames=30),
                                          **cam.to_json()}
            print(f"[render] {cam.name}: first-last mean |ΔRGB| = {mad:.2f}/255; "
                  f"steady {report['cameras'][cam.name]['profile']['steady_median_ms']:.2f} ms", flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"[render] checks: {json.dumps(checks)}", flush=True)
        print(f"[render] wrote {args.out.resolve()}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
