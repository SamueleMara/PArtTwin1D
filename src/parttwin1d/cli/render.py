#!/usr/bin/env python3
"""Render an existing articulated URDF + per-link 3DGS using real ManiSkill link poses.

This is a deterministic pose-sequence integration benchmark, *not* a replacement
for physics stepping, collision geometry, force learning, or PPO.

Usage (from PArtTwin1D repo root):
 python scripts/render_articulated.py --config configs/hinged_box.json \
     --joint door_hinge --q-start 0 --q-end 1.3 --frames 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Allow running directly in an existing git checkout: no pip editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parttwin1d.sim.assets import load_config
from parttwin1d.sim.env import create_articulated_scene, set_joint_qpos
from parttwin1d.sim.renderer import ArticulatedGaussianRenderer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path, help="URDF / Gaussian PLY / camera JSON")
    p.add_argument("--joint", help="Active URDF joint name; optional only for single-DOF objects")
    p.add_argument("--q-start", type=float, default=None)
    p.add_argument("--q-end", type=float, default=None)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("outputs/hinged_box"))
    p.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu")
    p.add_argument("--max-gaussians-per-link", type=int, default=0,
                   help="Only for performance troubleshooting; 0 uses all Gaussians")
    return p.parse_args()


def get_joint(art, name, qstart, qend):
    names = [j.name for j in art.get_active_joints()]
    if not names:
        raise ValueError("URDF has no movable joints")
    if name is None:
        if len(names) != 1:
            raise ValueError(f"Pass --joint; active joints: {names}")
        name = names[0]
    if name not in names:
        raise ValueError(f"Unknown joint {name!r}. Available active joints: {names}")
    i = names.index(name)
    lower, upper = art.qlimits[0, i, :].detach().cpu().tolist()
    if not np.isfinite([lower, upper]).all():
        raise ValueError(f"Joint {name}: finite URDF joint limits required for this sequence driver")
    return name, float(lower if qstart is None else qstart), float(upper if qend is None else qend), names


def run(args):
    if args.frames < 2:
        raise ValueError("--frames must be >= 2")
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    env = create_articulated_scene(config, args.sim_backend)
    try:
        env.reset(seed=0)
        art = env.twin
        name, qstart, qend, all_joints = get_joint(art, args.joint, args.q_start, args.q_end)
        print(f"Imported {config.urdf.name} | links={sorted(art.links_map)} | joints={all_joints}", flush=True)
        renderer = ArticulatedGaussianRenderer(config, max_gaussians_per_link=args.max_gaussians_per_link)
        renderer.validate_links(art, config.links)
        print(f"Loaded {renderer.gaussian_count} Gaussians across {len(config.links)} URDF links", flush=True)
        report = {"config": str(config.source), "urdf": str(config.urdf), "joint": name,
                  "q_start": qstart, "q_end": qend, "frames": args.frames,
                  "sim_backend": args.sim_backend, "gaussians": renderer.gaussian_count,
                  "camera_convention": "T_world_camera, OpenCV optical", "sequence": []}
        all_frames = {camera.name: [] for camera in config.cameras}
        q_values = np.linspace(qstart, qend, args.frames)
        for frame_idx, q in enumerate(q_values):
            qpos = set_joint_qpos(env, name, float(q))
            fdata = {"index": frame_idx, "qpos": qpos, "cameras": {}}
            for camera in config.cameras:
                result = renderer.render_numpy(art, camera)
                image = Image.fromarray(result["rgb"], "RGB")
                img_dir = args.out / camera.name
                img_dir.mkdir(exist_ok=True, parents=True)
                image.save(img_dir / f"rgb_{frame_idx:04d}.png")
                np.savez_compressed(img_dir / f"depth_alpha_{frame_idx:04d}.npz",
                                    depth_m=result["depth_m"], alpha=result["alpha"])
                all_frames[camera.name].append(image)
                fdata["cameras"][camera.name] = {
                    "alpha_pixels": int((result["alpha"] > 0.01).sum()),
                    "T_world_ply": result["T_world_ply"],
                }
            report["sequence"].append(fdata)
            print(f"frame {frame_idx+1}/{len(q_values)} joint={q:.5f}", flush=True)
        for camera in config.cameras:
            frames = all_frames[camera.name]
            frames[0].save(args.out / f"{camera.name}.gif", save_all=True,
                           append_images=frames[1:], duration=120, loop=0)
            first, last = (np.asarray(frames[0], np.float32), np.asarray(frames[-1], np.float32))
            mad = float(np.abs(first-last).mean())
            report.setdefault("image_mean_absolute_change", {})[camera.name] = mad
            print(f"{camera.name}: first-last mean RGB absolute change = {mad:.3f}", flush=True)
            if mad < 1.0:
                print(f"WARNING: {camera.name} barely changed; check joint-to-PLY binding and camera", flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Rendered RGB, alpha/depth, per-link poses and GIF to {args.out.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    run(parse_args())
