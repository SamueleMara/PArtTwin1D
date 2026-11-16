#!/usr/bin/env python3
"""Minimal programmatic use of the production API (no GUI).

    python examples/demo_articulated.py --config configs/examples/hinged_box.yaml

Loads the URDF through ManiSkill (real PhysX articulation), binds the per-link
Gaussians, sets a few joint values kinematically, renders the first calibrated
camera with gsplat and writes RGB/depth/alpha + metadata to outputs/demo/.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/examples/hinged_box.yaml")
    ap.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = ap.parse_args()

    from parttwin1d.config import load_config
    from parttwin1d.rendering.cameras import RenderCamera
    from parttwin1d.rendering.gsplat_backend import ArticulatedGaussianRenderer
    from parttwin1d.rendering.observations import export_result
    from parttwin1d.sim.articulation import ArticulationDriver
    from parttwin1d.sim.env import create_env

    cfg = load_config(args.config)
    env = create_env(cfg)                      # robot-free ManiSkill scene, render_backend="none"
    try:
        env.reset(seed=0)
        driver = ArticulationDriver(env, cfg)  # active joints discovered by name from PhysX
        renderer = ArticulatedGaussianRenderer.from_config(cfg)
        renderer.bind_articulation(env.twin)
        camera = RenderCamera.from_spec(cfg.cameras[0])
        joint = driver.joints[0]
        for q in (joint.lower, 0.5 * (joint.lower + joint.upper), joint.upper):
            driver.set_joint_kinematic(joint.name, q)       # kinematic preview (no time passes)
            snapshot = driver.snapshot("kinematic")          # immutable q + T_world_link
            renderer.update_from_snapshot(snapshot)
            result = renderer.render(camera, segmentation=True).validate()  # GPU tensors
            paths = export_result(result, args.out, f"{joint.name}_{q:+.3f}")
            print(f"{joint.name}={q:+.3f} {joint.unit}: {int(result.valid.sum())} visible px, "
                  f"{result.render_ms:.1f} ms -> {paths['rgb']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
