#!/usr/bin/env python3
"""Capture GUI evidence screenshots in a real (headless) Chrome via Playwright.

Starts the live GUI in-process, connects a browser, and saves full-page
screenshots of the *live* gsplat render at q = q0 and q = q_max-ish in RGB,
depth and link-ID modes (same orbit camera), plus a JSON log of render events.

    python scripts/capture_gui_evidence.py --config configs/examples/hinged_box.yaml
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _settle(app, quiet=0.8, timeout=20):
    t_end = time.time() + timeout
    while time.time() < t_end:
        n = app.render_cb.count
        time.sleep(quiet)
        if n and app.render_cb.count == n:
            return
    raise TimeoutError("renders did not settle")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/examples/hinged_box.yaml")
    ap.add_argument("--joint", default=None)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/evidence"))
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    from parttwin1d.config import load_config
    from parttwin1d.viewer.server import GuiApp

    args.out.mkdir(parents=True, exist_ok=True)
    app = GuiApp(load_config(args.config), port=_free_port(), out_dir=args.out / "exports", verbose=False)
    joint = args.joint or app.coordinator.joints()[0].name
    shots = {}
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True,
                                  args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": 1500, "height": 900})
            pg.goto(app.url, wait_until="networkidle")
            init = np.asarray(app.server.initial_camera.position)
            t_end = time.time() + 30
            while time.time() < t_end:
                e = app.render_cb.latest()
                if e is not None and np.allclose(np.array(e.c2w).reshape(4, 4)[:3, 3], init, atol=1e-3):
                    break
                time.sleep(0.05)
            for btn in pg.get_by_role("button").all():  # close viser's software-WebGL toast if present
                if btn.get_attribute("class") and "CloseButton" in (btn.get_attribute("class") or ""):
                    btn.click()
            for q in (0.0, args.q):
                rid = app.coordinator.submit_joint(joint, q)
                app.coordinator.wait_for(lambda s: s.request_id >= rid)
                for mode in ("rgb", "depth", "link_id") if q else ("rgb",):
                    app.render_cb.options.mode = mode
                    app.rerender_all()
                    _settle(app)
                    pg.wait_for_timeout(700)
                    name = f"gui_{mode}_q{q:.2f}.png"
                    pg.screenshot(path=str(args.out / name))
                    ev = app.render_cb.latest()
                    shots[name] = {"q": q, "mode": mode, "revision": ev.revision, "width": ev.width,
                                   "height": ev.height, "render_ms": ev.ms, "K": ev.K, "c2w": ev.c2w}
            app.render_cb.options.mode = "rgb"
            b.close()
        (args.out / "gui_evidence.json").write_text(json.dumps(shots, indent=2))
        print(f"wrote {len(shots)} screenshots to {args.out.resolve()}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
