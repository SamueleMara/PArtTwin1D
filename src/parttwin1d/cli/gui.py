"""Launch the live interactive 3D Gaussian GUI.

    python -m parttwin1d.cli.gui --config configs/examples/hinged_box.yaml --port 8080
    # open http://127.0.0.1:8080

Remote machine: keep the default 127.0.0.1 binding and forward the port:
    ssh -N -L 8080:127.0.0.1:8080 user@workstation
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parttwin gui", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="object config (.yaml/.json)")
    p.add_argument("--host", default=None, help="bind address (default from config: 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port (default from config: 8080)")
    p.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default=None)
    p.add_argument("--max-res", type=int, default=None, help="max live render resolution (longest side)")
    p.add_argument("--out", default="outputs", help="export root; files go to <out>/<run_id>/")
    p.add_argument("--run-id", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from ..config import load_config
    from ..viewer.server import GuiApp

    cfg = load_config(args.config)
    host = args.host or cfg.viewer.host
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {host!r} exposes the GUI beyond this machine.", file=sys.stderr)
    app = GuiApp(cfg, host=host, port=args.port, sim_backend=args.sim_backend, max_res=args.max_res,
                 out_dir=args.out, run_id=args.run_id)
    requested = args.port or cfg.viewer.port
    if app.port != requested:
        print(f"WARNING: port {requested} is in use; viser bound {app.port} instead.", file=sys.stderr, flush=True)
    print(f"[parttwin gui] {cfg.object_name}: {app.renderer.gaussian_count} Gaussians, "
          f"links {list(app.renderer.link_names)}, joints {[j.name for j in app.coordinator.joints()]}", flush=True)
    print(f"[parttwin gui] open {app.url}  (Ctrl+C to stop; exports -> {app.out_dir})", flush=True)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
