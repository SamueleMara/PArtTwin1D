"""Open ManiSkill's own SAPIEN viewer with the per-link Gaussians composited in.

    python -m parttwin1d.cli.maniskill_gui --config configs/examples/hinged_box.yaml

Native desktop window (needs an X/Wayland display and Vulkan). The Gaussians are
rendered by gsplat from the SAPIEN viewer camera every frame and depth-composited
with the SAPIEN scene in the same frame (no streaming, no lag). Mouse: right-drag
rotate, middle/shift-drag pan, wheel zoom (SAPIEN viewer controls). The
"PArtTwin1D Gaussian twin" window has joint sliders, sweep/physics, cameras,
export; SAPIEN's own Articulation window also drives the joints.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="parttwin maniskill-gui", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--render-scale", type=float, default=1.0, help="gsplat layer resolution relative to the window")
    p.add_argument("--no-collision-proxies", action="store_true", help="do not add collision-shape debug visuals")
    p.add_argument("--sim-backend", default="physx_cpu", choices=("physx_cpu",))
    p.add_argument("--out", default="outputs")
    p.add_argument("--frames", type=int, default=None, help="exit after N frames (automation)")
    p.add_argument("--screenshot", default=None, help="with --frames: save the composited viewer frame")
    args = p.parse_args(argv)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("ERROR: no display (DISPLAY/WAYLAND_DISPLAY unset). Use `parttwin gui` (browser) on headless machines.",
              file=sys.stderr)
        return 2
    from ..config import load_config
    from ..viewer.maniskill_viewer import ManiSkillGaussianViewer

    cfg = load_config(args.config)
    app = ManiSkillGaussianViewer(cfg, width=args.width, height=args.height, render_scale=args.render_scale,
                                  collision_proxies=not args.no_collision_proxies, out_dir=args.out,
                                  sim_backend=args.sim_backend)
    print(f"[parttwin maniskill-gui] {cfg.object_name}: {app.renderer.gaussian_count} Gaussians in the ManiSkill "
          f"viewer (shader pack {app.shader_pack}); close the window to exit", flush=True)
    app.run(max_frames=args.frames, screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
