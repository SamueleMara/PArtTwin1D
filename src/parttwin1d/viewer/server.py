"""Live interactive 3D GUI host: viser + nerfview (pinned) + gsplat + ManiSkill.

Threads
-------
* simulation owner  (``SimulationCoordinator``): the only thread touching ManiSkill.
* nerfview renderer (one per browser client): camera state -> ``render_fn``.
* calibrated view worker: exact-K render of the selected real camera.
* status ticker: FPS / status text every 0.5 s (blocks on an Event, no busy loop).
* viser's own event loop / callback pool for GUI events.

Shutdown (Ctrl+C / SIGTERM / ``close()``): wait for in-flight exports, stop the
workers, stop the simulation owner (which closes the ManiSkill scene in its own
thread), then stop the viser server.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..assets.urdf import parse_urdf
from ..rendering.cameras import RenderCamera
from ..rendering.gsplat_backend import ArticulatedGaussianRenderer
from ..rendering.observations import export_result, rgb_uint8
from .coordination import SimulationCoordinator
from .overlays import OverlayManager
from .render_callback import GuiRenderOptions, ViewerRenderCallback


def _make_viewer_class():
    import nerfview

    class ParttwinRenderTabState(nerfview.RenderTabState):
        pass

    class ParttwinViewer(nerfview.Viewer):
        """nerfview Viewer using our own control panel instead of the generic
        render tab (camera-path/video tooling is out of scope)."""

        def __init__(self, server, render_fn, output_dir, max_res: int):
            self._max_res = int(max_res)
            super().__init__(server, render_fn, output_dir=output_dir, mode="rendering")

        def _init_rendering_tab(self):
            self.render_tab_state = ParttwinRenderTabState()
            self.render_tab_state.viewer_res = self._max_res
            self._rendering_tab_handles = {}
            self._rendering_folder = None

        def _populate_rendering_tab(self):
            pass

        def safe_rerender(self):
            """Like nerfview.Viewer.rerender, tolerant of connect/disconnect races."""
            from nerfview._renderer import RenderTask

            for cid, client in list(self.server.get_clients().items()):
                r = self._renderers.get(cid)
                if r is None:
                    continue
                try:
                    r.submit(RenderTask("rerender", self.get_camera_state(client)))
                except Exception:
                    pass

    return ParttwinViewer


class GuiApp:
    def __init__(self, cfg, *, host: str | None = None, port: int | None = None, sim_backend: str | None = None,
                 max_res: int | None = None, out_dir: str | Path = "outputs", run_id: str | None = None,
                 device: str = "cuda", verbose: bool = True):
        import viser

        from ..sim.articulation import ArticulationDriver
        from ..sim.env import create_env

        self.cfg = cfg
        self.verbose = verbose
        self.run_id = run_id or time.strftime(f"gui_{cfg.object_name}_%Y%m%d_%H%M%S")
        self.out_dir = Path(out_dir).resolve() / self.run_id
        self._closed = False
        self._stop_evt = threading.Event()
        self._exports_inflight = 0
        self._exports_cv = threading.Condition()
        self.urdf = parse_urdf(cfg.urdf)

        backend = sim_backend or cfg.sim.backend

        def factory():
            env = create_env(cfg, backend)
            env.reset(seed=0)
            return ArticulationDriver(env, cfg)

        self.coordinator = SimulationCoordinator(factory, playback_hz=cfg.viewer.playback_hz)
        self.coordinator.start()
        self.renderer = ArticulatedGaussianRenderer.from_config(cfg, device=device)
        self.renderer.bind_articulation(self.coordinator.snapshot().T_world_link.keys())
        self.renderer.update_from_snapshot(self.coordinator.snapshot())

        self.server = viser.ViserServer(host=host or cfg.viewer.host, port=port or cfg.viewer.port,
                                        label=f"PArtTwin1D {cfg.object_name}", verbose=False)
        self.server.scene.set_up_direction("+z")
        c, r = self.object_center(), self.object_radius()
        self.server.initial_camera.look_at = tuple(c)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.position = tuple(c + np.array([1.0, -1.4, 0.9]) / np.linalg.norm([1.0, -1.4, 0.9]) * 3.2 * r)

        self.render_cb = ViewerRenderCallback(self.renderer, self.coordinator, GuiRenderOptions(),
                                              on_error=lambda src, exc, msg: self.report_error(src, msg))
        Viewer = _make_viewer_class()
        self.viewer = Viewer(self.server, self.render_cb, self.out_dir, max_res or cfg.viewer.max_resolution)
        cams = [RenderCamera.from_spec(c) for c in cfg.cameras]
        self.overlays = OverlayManager(self.server, self.urdf, cfg.links, self.renderer.scene.canonical, cams)
        self.overlays.update(self.coordinator.snapshot())

        from .controls import GuiControls

        self.controls = GuiControls(self)
        self.coordinator.subscribe(self._on_snapshot)
        self.coordinator.subscribe_errors(lambda ev: self.report_error(ev.source, ev.message))

        self._calib_evt = threading.Event()
        self._calib_thread = threading.Thread(target=self._calibrated_worker, name="parttwin-calibrated", daemon=True)
        self._calib_thread.start()
        self._status_thread = threading.Thread(target=self._status_worker, name="parttwin-status", daemon=True)
        self._status_thread.start()
        self.request_calibrated_render()

    # ----------------------------------------------------------- properties
    @property
    def port(self) -> int:
        return int(self.server.get_port())

    @property
    def url(self) -> str:
        host = self.server.get_host()
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{self.port}"

    def object_center(self) -> np.ndarray:
        m = self.renderer.world_means()
        return ((m.min(0).values + m.max(0).values) / 2).detach().cpu().numpy().astype(np.float64)

    def object_radius(self) -> float:
        m = self.renderer.world_means()
        return float(((m.max(0).values - m.min(0).values) / 2).norm().item()) or 0.5

    # ----------------------------------------------------------- events
    def _on_snapshot(self, snap):
        # Called in the simulation owner thread after each publish.
        self.overlays.update(snap)
        self.controls.sync_joints(snap)
        self.controls.sync_dynamics(snap)
        self.viewer.safe_rerender()
        self.request_calibrated_render()

    def report_error(self, source: str, exc_or_msg) -> None:
        msg = exc_or_msg if isinstance(exc_or_msg, str) else f"{type(exc_or_msg).__name__}: {exc_or_msg}"
        print(f"[parttwin gui] ERROR {source}: {msg}", file=sys.stderr, flush=True)
        try:
            self.controls.show_error(source, msg)
        except Exception:
            pass

    def rerender_all(self) -> None:
        self.viewer.safe_rerender()

    def request_calibrated_render(self) -> None:
        self._calib_evt.set()

    # ----------------------------------------------------------- workers
    def _calibrated_worker(self):
        last_key = None
        while not self._stop_evt.is_set():
            if not self._calib_evt.wait(0.5):
                continue
            self._calib_evt.clear()
            try:
                ctl = self.controls
                spec = ctl.selected_calibrated()
                if spec is None or not ctl.calib_enable.value:
                    continue
                snap = self.coordinator.snapshot()
                key = (spec.name, snap.revision, self.render_cb.options.sh_degree, ctl.r_bg.value)
                if key == last_key:
                    continue
                cam = RenderCamera.from_spec(spec)
                w = min(cam.width, 480)
                prev = cam.resized(w, round(cam.height * w / cam.width))
                with self.renderer._lock:
                    self.renderer.update_from_snapshot(snap)
                    res = self.renderer.render(prev, sh_degree=self.render_cb.options.sh_degree)
                ctl.calib_img.image = rgb_uint8(res.rgb.detach().cpu().numpy())
                K = cam.K
                ctl.calib_md.content = (
                    f"**{spec.name}** (calibrated, exact K) {cam.width}×{cam.height}: fx={K[0,0]:.2f} "
                    f"fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}; preview {prev.width}×{prev.height} "
                    f"with exactly rescaled K; rev {snap.revision}")
                last_key = key
            except Exception as exc:
                self.report_error("calibrated view", exc)

    def _status_worker(self):
        while not self._stop_evt.wait(0.5):
            try:
                self.controls.set_status()
                self.controls.update_camera_md()
            except Exception as exc:  # pragma: no cover
                self.report_error("status", exc)

    # ----------------------------------------------------------- exports
    def export_snapshot(self, client=None) -> Path:
        with self._exports_cv:
            self._exports_inflight += 1
        try:
            snap = self.coordinator.snapshot()
            out = self.out_dir / time.strftime("snapshot_%Y%m%d_%H%M%S") / f"rev{snap.revision:06d}"
            out.mkdir(parents=True, exist_ok=True)
            written = {}
            with self.renderer._lock:
                self.renderer.update_from_snapshot(snap)
                if client is not None:
                    cs = self.viewer.get_camera_state(client)
                    tab = self.viewer.render_tab_state
                    cam = RenderCamera.virtual(cs.c2w, cs.fov, tab.viewer_width, tab.viewer_height,
                                               name=f"orbit_client{client.client_id}")
                    res = self.renderer.render(cam, segmentation=True)
                    written[cam.name] = str(export_result(res, out, cam.name, {"camera_kind": "virtual orbit K"})["meta"])
                for spec in self.cfg.cameras:
                    cam = RenderCamera.from_spec(spec)
                    res = self.renderer.render(cam, segmentation=True)
                    written[cam.name] = str(export_result(res, out, cam.name, {"camera_kind": "calibrated exact K"})["meta"])
            (out / "state.json").write_text(json.dumps({"snapshot": snap.to_json(), "files": written,
                                                         "config": str(self.cfg.source)}, indent=2))
            return out
        finally:
            with self._exports_cv:
                self._exports_inflight -= 1
                self._exports_cv.notify_all()

    def export_trajectory(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / time.strftime("trajectory_%Y%m%d_%H%M%S.json")
        names = [j.name for j in self.coordinator.joints()]
        rows = [{"sim_time": t, "wall_time": w, "revision": r, "mode": m, "q": dict(zip(names, q))}
                for t, w, r, m, q in list(self.coordinator.history)]
        p.write_text(json.dumps({"joints": names, "samples": rows}, indent=1))
        return p

    # ----------------------------------------------------------- lifecycle
    def run(self) -> None:
        """Block until Ctrl+C / SIGTERM / close()."""
        def handler(signum, frame):
            self._stop_evt.set()
        prev = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not self._stop_evt.wait(0.5):
                if not self.coordinator.alive:
                    self.report_error("simulation", "simulation owner thread exited")
                    break
        finally:
            for s, h in prev.items():
                signal.signal(s, h)
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._exports_cv:
            self._exports_cv.wait_for(lambda: self._exports_inflight == 0, timeout=30)
        self._stop_evt.set()
        self._calib_evt.set()
        for t in (self._calib_thread, self._status_thread):
            t.join(5)
        # Stop nerfview's per-client render threads and WAIT for them: a daemon
        # thread still inside a CUDA call at interpreter exit aborts the process
        # ("terminate called without an active exception").
        # Includes threads of already-disconnected clients (nerfview drops them
        # from _renderers without joining).
        from nerfview._renderer import Renderer as _NerfviewRenderer

        renderers = [t for t in threading.enumerate() if isinstance(t, _NerfviewRenderer) and t.viewer is self.viewer]
        for r in renderers:
            r.running = False
            r._render_event.set()
        for r in renderers:
            r.join(5)
        self.coordinator.stop()
        self.server.stop()
        if self.verbose:
            print("[parttwin gui] shut down cleanly", flush=True)
