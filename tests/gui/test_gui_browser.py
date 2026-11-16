"""REAL browser GUI tests: system Google Chrome (headless) via Playwright, live viser
server, pinned nerfview, real gsplat CUDA renders and real ManiSkill PhysX.

Opt-in:  PARTTWIN_RUN_GUI_TESTS=1 python -m pytest -v tests/gui
Screenshots are written to outputs/test_artifacts/gui_*.png as evidence.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from tests.conftest import HINGED, ROOT, requires_gui

pytestmark = [requires_gui, pytest.mark.gui]
ART = ROOT / "outputs" / "test_artifacts"
CHROME_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_until(fn, timeout=20.0, interval=0.05, msg="condition"):
    t_end = time.time() + timeout
    last = None
    while time.time() < t_end:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timeout waiting for {msg} (last={last!r})")


def red_door_pixels(png_path) -> int:
    from PIL import Image

    a = np.asarray(Image.open(png_path).convert("RGB")).astype(int)[:, :1000]  # exclude GUI panel
    return int(((a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] < 110)).sum())


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from parttwin1d.config import load_config
    from parttwin1d.viewer.server import GuiApp

    a = GuiApp(load_config(HINGED), port=free_port(), out_dir=tmp_path_factory.mktemp("gui_out"), verbose=False)
    yield a
    a.close()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=CHROME_ARGS)
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(app, browser):
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    pg = ctx.new_page()
    pg.goto(app.url, wait_until="networkidle")
    init = np.asarray(app.server.initial_camera.position)
    # the browser first reports viser's default camera, then applies the configured initial camera
    wait_until(lambda: (e := app.render_cb.latest()) is not None
               and np.allclose(np.array(e.c2w).reshape(4, 4)[:3, 3], init, atol=1e-3), msg="render at initial camera")
    settle(app)
    yield pg
    ctx.close()


def settle(app, quiet_s=0.8, timeout=20.0):
    """Wait until nerfview stops producing renders (camera transition + low->high res done)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        n = app.render_cb.count
        time.sleep(quiet_s)
        if app.render_cb.count == n and n > 0:
            return app.render_cb.latest()
    raise AssertionError("renders did not settle")


def latest_for(app, cid=None):
    evs = [e for e in list(app.render_cb.log) if cid is None or e.client_hint == str(cid)]
    return evs[-1] if evs else None


def client_ids(app):
    return sorted(app.server.get_clients().keys())


def joint_slider(pg):
    sid = pg.locator("label", has_text=re.compile(r"hinge \[rad\]")).first.get_attribute("for")
    root = pg.locator(f"[id='{sid}']")
    return root, root.locator("xpath=..").get_by_role("textbox").first, root.get_by_role("slider")


def set_joint_via_box(pg, value: float):
    _, box, _ = joint_slider(pg)
    box.fill(f"{value}")
    box.press("Enter")
    box.evaluate("e => e.blur()")


def open_folder(pg, title: str, exact: bool = False):
    pg.get_by_text(title, exact=exact).locator("visible=true").first.click()
    time.sleep(0.3)


def screenshot(pg, name, app=None):
    ART.mkdir(parents=True, exist_ok=True)
    if app is not None:
        settle(app)
    pg.wait_for_timeout(600)
    p = ART / f"gui_{name}.png"
    pg.screenshot(path=str(p))
    return p


# --------------------------------------------------------------------------- tests
def test_live_gaussian_rgb_in_browser(app, page):
    ev = latest_for(app)
    assert ev.width > 100 and ev.height > 100 and ev.visible_px > 1000
    p = screenshot(page, "initial", app)
    assert red_door_pixels(p) > 2000, "browser does not show the gsplat-rendered red door"
    assert page.locator("canvas").count() >= 1
    assert "Shared articulation q" in page.content()


def test_orbit_pan_zoom_trigger_new_gsplat_renders(app, page):
    cid = client_ids(app)[0]
    box = page.locator("canvas").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    seen = []
    for action in ("orbit", "pan", "zoom"):
        before = latest_for(app, cid)
        n0 = app.render_cb.count
        if action == "orbit":
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + 160, cy + 40, steps=12)
            page.mouse.up()
        elif action == "pan":
            page.mouse.move(cx, cy)
            page.mouse.down(button="right")
            page.mouse.move(cx + 80, cy - 30, steps=8)
            page.mouse.up(button="right")
        else:
            page.mouse.move(cx, cy)
            page.mouse.wheel(0, -400)
        ev = wait_until(lambda: (e := latest_for(app, cid)) is not None and app.render_cb.count > n0
                        and not np.allclose(e.c2w, before.c2w, atol=1e-6) and e, msg=f"{action} render")
        seen.append(np.array(ev.c2w).reshape(4, 4))
        assert ev.revision == before.revision  # camera-only change: same physical state
    assert len({tuple(np.round(s.ravel(), 6)) for s in seen}) == 3
    page.set_viewport_size({"width": 1200, "height": 800})
    n0 = app.render_cb.count
    ev = wait_until(lambda: app.render_cb.count > n0 and latest_for(app, cid), msg="resize render")
    K = np.array(latest_for(app, cid).K).reshape(3, 3)
    assert K[0, 0] == K[1, 1] and K[0, 2] == latest_for(app, cid).width / 2  # documented virtual K
    page.set_viewport_size({"width": 1400, "height": 900})
    screenshot(page, "after_orbit", app)


def test_camera_motion_latency(app, page):
    """Orbit responsiveness: camera update -> frame sent, measured server-side during a real drag.
    Regression guard for the CPU post-processing / per-frame depth-PNG cost (was ~54 ms median)."""
    import statistics

    cid = client_ids(app)[0]
    client = app.server.get_clients()[cid]
    cam_t, lat, res = [], [], []
    client.camera.on_update(lambda c: cam_t.append(time.perf_counter()))
    orig = client.scene.set_background_image

    def timed(*a, **k):
        t = time.perf_counter()
        orig(*a, **k)
        now = time.perf_counter()
        pend = [c for c in cam_t if c < t]
        if pend:
            lat.append((now - min(pend)) * 1e3)
            cam_t[:] = [c for c in cam_t if c >= t]
        res.append(a[0].shape[:2])

    client.scene.set_background_image = timed
    try:
        box = page.locator("canvas").first.bounding_box()
        cx, cy = box["x"] + 400, box["y"] + 400
        page.mouse.move(cx, cy)
        page.mouse.down()
        for i in range(80):
            page.mouse.move(cx + 250 * np.sin(i / 15), cy + 50 * np.cos(i / 12))
            time.sleep(1 / 60)
        page.mouse.up()
        settle(app)
    finally:
        del client.scene.set_background_image
    assert len(lat) > 20, len(lat)
    med = statistics.median(lat)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "gui_latency.json").write_text(json.dumps({"camera_to_frame_ms_median": med,
                                                     "p90": sorted(lat)[int(0.9 * len(lat))], "frames": len(lat),
                                                     "resolutions": sorted({tuple(r) for r in res})}, indent=2))
    assert med < 30.0, f"camera->frame latency median {med:.1f} ms"
    assert max(h for h, w in res) >= 700  # no resolution collapse while moving


def test_joint_slider_moves_hinge_without_drift(app, page):
    from parttwin1d.evaluation.consistency import expected_child_pose, pose_error

    snap0 = app.coordinator.snapshot()
    T_base0 = snap0.T_world_link["base_link"]
    T_door0 = snap0.T_world_link["door_link"]
    uj = app.urdf.joints["door_hinge"]
    _, _, thumb = joint_slider(page)
    for target in (1.0, 0.0, 1.0):
        set_joint_via_box(page, target)
        s = wait_until(lambda: (x := app.coordinator.snapshot()) and abs(x.q[0] - target) < 1e-4 and x, msg=f"q={target}")
        wait_until(lambda: (e := latest_for(app)) and e.revision >= s.revision, msg="render of new revision")
        tr, rot = pose_error(expected_child_pose(T_door0, uj, target, float(snap0.q[0])), s.T_world_link["door_link"])
        assert rot < 1e-4 and tr < 1e-5
        assert max(pose_error(T_base0, s.T_world_link["base_link"])) < 1e-9
        wait_until(lambda: abs(float(thumb.get_attribute("aria-valuenow")) - target) < 1e-3, msg="slider sync")
        assert f"{target:+.4f} rad" in page.content()
        node = app.overlays.link_nodes["door_link"]  # overlays follow the SAPIEN link pose
        np.testing.assert_allclose(node.position, s.T_world_link["door_link"][:3, 3], atol=1e-6)
        if target == 1.0:
            screenshot(page, "q1", app)
    # real mouse drag on the slider track
    root, _, _ = joint_slider(page)
    tb = root.locator(".mantine-Slider-track").bounding_box()
    page.mouse.move(tb["x"] + tb["width"] * 0.05, tb["y"] + tb["height"] / 2)
    page.mouse.down()
    page.mouse.move(tb["x"] + tb["width"] * 0.32, tb["y"] + tb["height"] / 2, steps=15)
    page.mouse.up()
    s = wait_until(lambda: (x := app.coordinator.snapshot()) and 0.35 < x.q[0] < 0.65 and x,
                   msg=f"drag q (now {app.coordinator.snapshot().q})")
    set_joint_via_box(page, 0.0)
    s = wait_until(lambda: (x := app.coordinator.snapshot()) and abs(x.q[0]) < 1e-6 and x, msg="back to 0")
    assert max(pose_error(T_door0, s.T_world_link["door_link"])) < 1e-6  # no accumulated drift


def test_play_pause_respects_q(app, page):
    open_folder(page, "Kinematic sweep playback")
    r0 = app.coordinator.snapshot().revision
    page.get_by_role("button", name="Play").click()
    wait_until(lambda: app.coordinator.snapshot().revision > r0 + 10 and app.coordinator.snapshot().q[0] > 0.2,
               msg="sweep advancing")
    page.get_by_role("button", name="Pause").click()
    wait_until(lambda: not app.coordinator.playback.playing, msg="paused")
    q_a = app.coordinator.snapshot().q[0]
    time.sleep(0.6)
    assert app.coordinator.snapshot().q[0] == q_a
    page.get_by_role("button", name="Step").click()
    wait_until(lambda: app.coordinator.snapshot().q[0] != q_a, msg="single step")
    set_joint_via_box(page, 0.0)
    wait_until(lambda: abs(app.coordinator.snapshot().q[0]) < 1e-6, msg="reset q")


def test_overlay_toggle_keeps_splats(app, page):
    open_folder(page, "Overlays (debug")
    h = app.overlays.handles["link_frames"][0]
    assert h.visible
    page.get_by_label("Link frames").click()
    wait_until(lambda: not h.visible, msg="link frames hidden")
    p = screenshot(page, "overlays_off", app)
    assert red_door_pixels(p) > 2000  # Gaussians still rendered
    page.get_by_label("Link frames").click()
    wait_until(lambda: h.visible, msg="link frames shown")


def test_gui_errors_are_surfaced(app, page):
    open_folder(page, "Physics (dynamics preview)")
    page.get_by_label("Run physics").click()
    wait_until(lambda: "switch to dynamics mode" in page.content(), msg="error text in GUI")
    assert app.coordinator.alive


def test_dynamics_mode_from_gui(app, page):
    page.locator("label", has_text=re.compile(r"^Mode$")).first.click()
    page.get_by_role("option", name="dynamics").click()
    wait_until(lambda: app.coordinator.mode == "dynamics", msg="dynamics mode")
    page.get_by_role("button", name="Step physics ×10").click()
    wait_until(lambda: app.coordinator.snapshot().sim_time > 0.15, msg="physics advanced")
    s = app.coordinator.snapshot()
    assert s.mode == "dynamics"
    app.coordinator.set_mode("kinematic")
    wait_until(lambda: app.coordinator.mode == "kinematic")


def test_export_writes_K_c2w_q_links(app, page):
    open_folder(page, "Export")
    set_joint_via_box(page, 0.75)
    wait_until(lambda: abs(app.coordinator.snapshot().q[0] - 0.75) < 1e-4)
    before = set(app.out_dir.rglob("state.json")) if app.out_dir.exists() else set()
    page.get_by_role("button", name=re.compile("Save snapshot")).click()
    st = wait_until(lambda: (n := set(app.out_dir.rglob("state.json")) - before) and n, msg="export files")
    d = json.loads(next(iter(st)).read_text())
    assert d["snapshot"]["q"]["door_hinge"] == pytest.approx(0.75, abs=1e-4)
    assert set(d["files"]) >= {"front_oblique", "left_high"} and any(k.startswith("orbit_client") for k in d["files"])
    for name, meta in d["files"].items():
        md = json.loads(open(meta).read())
        assert np.asarray(md["camera"]["K"]).shape == (3, 3)
        assert np.asarray(md["camera"]["T_world_camera_c2w_opencv"]).shape == (4, 4)
        assert set(md["T_world_link"]) == {"base_link", "door_link"}
        for k in ("rgb", "alpha", "depth"):
            assert os.path.exists(os.path.join(os.path.dirname(meta), md["files"][k]))
    cal = json.loads(open(d["files"]["front_oblique"]).read())
    np.testing.assert_allclose(cal["camera"]["K"], [[600, 0, 331.5], [0, 590, 236], [0, 0, 1]])


def test_calibrated_view_panel_uses_exact_K(app, page):
    open_folder(page, "Camera", exact=True)
    wait_until(lambda: "exact K" in app.controls.calib_md.content and "fx=600.00" in app.controls.calib_md.content,
               msg="calibrated panel")
    assert app.controls.calib_img.image.shape[1] == 480  # preview width, K rescaled exactly


def test_two_clients_shared_q_independent_cameras(app, browser, page):
    ctx2 = browser.new_context(viewport={"width": 1100, "height": 800})
    p2 = ctx2.new_page()
    p2.goto(app.url, wait_until="networkidle")
    wait_until(lambda: len(client_ids(app)) == 2, msg="two clients")
    c1, c2 = client_ids(app)
    wait_until(lambda: latest_for(app, c2), msg="client 2 rendered")
    cam2_before = app.viewer.get_camera_state(app.server.get_clients()[c2]).c2w.copy()
    box = page.locator("canvas").first.bounding_box()
    page.mouse.move(box["x"] + 400, box["y"] + 400)
    page.mouse.down()
    page.mouse.move(box["x"] + 250, box["y"] + 380, steps=10)
    page.mouse.up()
    time.sleep(0.5)
    np.testing.assert_allclose(app.viewer.get_camera_state(app.server.get_clients()[c2]).c2w, cam2_before)
    # q changed from client 2 is seen by client 1 (shared articulation)
    set_joint_via_box(p2, 1.2)
    s = wait_until(lambda: (x := app.coordinator.snapshot()) and abs(x.q[0] - 1.2) < 1e-4 and x, msg="q from client 2")
    _, _, thumb1 = joint_slider(page)
    wait_until(lambda: abs(float(thumb1.get_attribute("aria-valuenow")) - 1.2) < 1e-3, msg="client 1 slider synced")
    wait_until(lambda: (e := latest_for(app, c1)) and e.revision >= s.revision, msg="client 1 re-rendered")
    wait_until(lambda: (e := latest_for(app, c2)) and e.revision >= s.revision, msg="client 2 re-rendered")
    screenshot(p2, "client2_q1.2", app)
    # disconnect / reconnect
    ctx2.close()
    wait_until(lambda: len(client_ids(app)) == 1 and len(app.viewer._renderers) == 1, msg="client 2 removed")
    ctx3 = browser.new_context(viewport={"width": 1000, "height": 700})
    p3 = ctx3.new_page()
    p3.goto(app.url, wait_until="networkidle")
    wait_until(lambda: len(client_ids(app)) == 2, msg="reconnected")
    c3 = [c for c in client_ids(app) if c != c1][0]
    wait_until(lambda: latest_for(app, c3), msg="reconnected client rendered")
    ctx3.close()
    set_joint_via_box(page, 0.0)


def test_cli_ctrl_c_clean_shutdown(tmp_path, browser):
    port = free_port()
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, "-u", "-m", "parttwin1d.cli.gui", "--config", str(HINGED),
                             "--port", str(port), "--out", str(tmp_path)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    t_end = time.time() + 120
    while time.time() < t_end:
        line = proc.stdout.readline()
        lines.append(line)
        if "open http" in line:
            break
    assert any(f"127.0.0.1:{port}" in ln for ln in lines), "".join(lines)
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
    pg.wait_for_timeout(2500)  # connected client with a live renderer thread
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=60)
    ctx.close()
    assert proc.returncode == 0, out
    assert "shut down cleanly" in out
