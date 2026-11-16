"""GUI panel: joints, playback, dynamics, cameras, render options, overlays, export, status.

Shared vs per-client state: the articulation q is SHARED by all connected
browsers (one simulation); every browser has its own independent orbit
camera. This is stated in the Status panel.

Slider callbacks act only on user events (``event.client is not None``);
programmatic slider syncs from published snapshots fire callbacks with
``client=None`` in viser 1.1.1 and are ignored, so there is no feedback loop.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..geometry.cameras import vertical_fov_from_K
from ..geometry.se3 import quat_from_matrix
from .overlays import OVERLAY_KINDS
from .render_callback import RENDER_MODES

OVERLAY_LABELS = {
    "link_frames": "Link frames", "joint_axes": "Joint axes + hinge", "labels": "Labels (link, q)",
    "bounding_boxes": "Gaussian bounding boxes", "collision": "Collision geometry",
    "camera_frusta": "Calibrated camera frusta", "world_axes": "World axes",
    "debug_grid": "Debug grid (debug only, not 3DGS)",
}
OVERLAY_DEFAULTS = {"link_frames": True, "joint_axes": True, "labels": True, "bounding_boxes": False,
                    "collision": False, "camera_frusta": True, "world_axes": True, "debug_grid": False}


class GuiControls:
    def __init__(self, app):
        self.app = app
        self.server = app.server
        self.coord = app.coordinator
        self.sliders: dict = {}
        self.readouts: dict = {}
        self._last_user_input: dict = {}
        self._error_lines: list[str] = []
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible", dark_mode=False, brand_color=(200, 60, 50),
                            control_width="large")
        gui.set_panel_label(f"PArtTwin1D — {app.cfg.object_name}")
        self._build_status()
        self._build_joints()
        self._build_playback()
        self._build_dynamics()
        self._build_camera()
        self._build_render()
        self._build_overlays()
        self._build_export()

    # ------------------------------------------------------------- status
    def _build_status(self):
        gui = self.server.gui
        with gui.add_folder("Status", expand_by_default=True):
            self.status_md = gui.add_markdown("starting…")
            self.camera_md = gui.add_markdown("")
            self.error_md = gui.add_markdown("")

    def set_status(self, snap=None):
        app = self.app
        snap = snap or self.coord.snapshot()
        cb = app.render_cb
        ev = cb.latest()
        res = f"{ev.width}×{ev.height}" if ev else "–"
        lines = [
            f"**Shared articulation q** (all browsers); **per-browser camera**. "
            f"Clients: {len(self.server.get_clients())}",
            f"Mode: `{self.coord.mode}` · revision {snap.revision if snap else '–'} · "
            f"sim t = {snap.sim_time if snap else 0:.3f} s",
            f"Render: gsplat 1.5.3 CUDA · {app.renderer.gaussian_count} Gaussians · {res} · "
            f"last frame {ev.ms if ev else float('nan'):.1f} ms (≈{1000 / ev.ms if ev and ev.ms > 0 else 0:.0f} fps "
            f"capacity) · {cb.fps():.1f} renders/s (on demand)",
        ]
        if snap and snap.note:
            lines.append(f"_{snap.note}_")
        if ev is not None and ev.visible_px == 0:
            lines.append("⚠️ **No visible alpha** in the last frame: object outside the view?")
        self.status_md.content = "  \n".join(lines)

    def show_error(self, source: str, message: str):
        stamp = time.strftime("%H:%M:%S")
        self._error_lines = ([f"`{stamp}` **{source}**: {message}"] + self._error_lines)[:6]
        self.error_md.content = "**Errors / warnings**  \n" + "  \n".join(self._error_lines)

    # ------------------------------------------------------------- joints
    def _build_joints(self):
        gui = self.server.gui
        snap = self.coord.snapshot()
        with gui.add_folder("Joints (URDF active DoF)", expand_by_default=True):
            for j in self.coord.joints():
                lo, hi = (j.lower, j.upper) if j.bounded else (-math.pi, math.pi)
                step = 1e-3 if j.unit == "rad" else 1e-4
                q = float(snap.q[j.index])
                hint = (f"{j.type} joint, child link {j.child_link}; URDF limits "
                        f"[{j.lower:.4f}, {j.upper:.4f}] {j.unit}" if j.bounded
                        else f"{j.type} joint (unbounded; slider shows ±π)")
                s = gui.add_slider(f"{j.name} [{j.unit}]", min=lo, max=hi, step=step, initial_value=q, hint=hint)
                r = gui.add_markdown(self._readout(j, q))
                if not j.bounded:
                    self.show_error("joints", f"{j.name} is unbounded; sweep playback disabled for it")
                s.on_update(self._make_joint_cb(j.name))
                self.sliders[j.name] = s
                self.readouts[j.name] = r
            b = gui.add_button("Reset to q0", hint="Teleport all joints to the configured initial q")
            b.on_click(lambda e: self._safe(self.coord.reset))

    @staticmethod
    def _readout(j, q):
        status = "zero" if abs(q) < 1e-6 else "non-zero"
        lim = f"[{j.lower:.3f}, {j.upper:.3f}]" if j.bounded else "(unbounded)"
        return f"`{j.name}` q = **{q:+.4f} {j.unit}** · {status} · limits {lim}"

    def _make_joint_cb(self, name):
        def cb(event):
            if event.client is None:  # programmatic sync from a snapshot
                return
            self._last_user_input[name] = time.time()
            self._safe(self.coord.submit_joint, name, float(event.target.value))
        return cb

    def sync_joints(self, snap):
        for j in self.coord.joints():
            q = float(snap.q[j.index])
            self.readouts[j.name].content = self._readout(j, q)
            s = self.sliders[j.name]
            recent = time.time() - self._last_user_input.get(j.name, 0) < 0.5
            if not recent and abs(s.value - q) > 5e-4:
                s.value = round(q, 4)

    # ------------------------------------------------------------- playback
    def _build_playback(self):
        gui = self.server.gui
        names = [j.name for j in self.coord.joints() if j.bounded]
        with gui.add_folder("Kinematic sweep playback", expand_by_default=False):
            gui.add_markdown("Kinematic preview: q is **set**, not simulated. Manual slider input pauses the sweep.")
            self.pb_joint = gui.add_dropdown("Sweep joint", tuple(names) or ("–",), initial_value=names[0] if names else "–")
            row = gui.add_button_group("Playback", ("Play", "Pause", "Step"))
            self.pb_loop = gui.add_checkbox("Loop", True)
            self.pb_speed = gui.add_slider("Speed ×", min=0.1, max=4.0, step=0.1, initial_value=1.0)

            @row.on_click
            def _(e):
                v = e.target.value
                if v == "Play":
                    self._safe(self.coord.play, self.pb_joint.value)
                elif v == "Pause":
                    self._safe(self.coord.pause)
                else:
                    self._safe(self.coord.step_playback, 1)
            self.pb_loop.on_update(lambda e: e.client and self._safe(self.coord.set_loop, self.pb_loop.value))
            self.pb_speed.on_update(lambda e: e.client and self._safe(self.coord.set_speed, self.pb_speed.value))

    # ------------------------------------------------------------- dynamics
    def _build_dynamics(self):
        gui = self.server.gui
        cfg = self.app.cfg.sim
        names = [j.name for j in self.coord.joints()]
        with gui.add_folder("Physics (dynamics preview)", expand_by_default=False):
            gui.add_markdown(
                f"PhysX stepping with an **explicit external input**: PD joint drive "
                f"(k={cfg.drive_stiffness:g}, d={cfg.drive_damping:g}, |f|≤{cfg.drive_force_limit:g}) "
                f"or a generalized joint force. Gravity, URDF inertia/damping/friction simulated. "
                f"Sim {cfg.sim_freq} Hz / control {cfg.control_freq} Hz.")
            self.dyn_mode = gui.add_dropdown("Mode", ("kinematic", "dynamics"), initial_value="kinematic")
            self.dyn_run = gui.add_checkbox("Run physics", False)
            self.dyn_kind = gui.add_dropdown("Input", ("drive", "force"), initial_value="drive",
                                             hint="drive: PD target q*; force: generalized force/torque")
            self.dyn_joint = gui.add_dropdown("Input joint", tuple(names) or ("–",), initial_value=names[0] if names else "–")
            self.dyn_value = gui.add_number("Input value (rad|m or N·m|N)", initial_value=0.0, step=0.01)
            apply_b = gui.add_button("Apply input")
            step_b = gui.add_button("Step physics ×10")

            self.dyn_mode.on_update(lambda e: e.client and self._safe(self.coord.set_mode, self.dyn_mode.value))
            self.dyn_run.on_update(lambda e: e.client and self._safe(self.coord.set_physics_running, self.dyn_run.value))
            apply_b.on_click(lambda e: self._safe(self.coord.set_dynamics_input, self.dyn_kind.value,
                                                  self.dyn_joint.value, float(self.dyn_value.value)))
            step_b.on_click(lambda e: self._safe(self.coord.physics_step, 10))

    def sync_dynamics(self, snap):
        if self.dyn_mode.value != self.coord.mode:
            self.dyn_mode.value = self.coord.mode
        if self.dyn_run.value != self.coord.dynamics.running:
            self.dyn_run.value = self.coord.dynamics.running

    # ------------------------------------------------------------- cameras
    def _build_camera(self):
        gui = self.server.gui
        app = self.app
        names = ("free orbit",) + tuple(c.name for c in app.cfg.cameras)
        with gui.add_folder("Camera", expand_by_default=False):
            gui.add_markdown(
                "Orbit: drag · Pan: right-drag / shift-drag · Zoom: wheel. The free orbit view uses a "
                "**virtual** pinhole K (fx = fy from the browser FOV, centred principal point). "
                "Calibrated presets move the orbit camera to the sensor pose; the exact-K render is "
                "shown in the *Calibrated view* panel below.")
            self.cam_preset = gui.add_dropdown("Preset", names, initial_value="free orbit")
            go = gui.add_button("Go to preset pose")
            center = gui.add_button("Center on object")
            self.calib_enable = gui.add_checkbox("Show calibrated view (exact K)", bool(app.cfg.cameras))
            self.calib_md = gui.add_markdown("")
            blank = np.full((60, 80, 3), 255, np.uint8)
            self.calib_img = gui.add_image(blank, label="calibrated view", format="jpeg", jpeg_quality=90)

            @go.on_click
            def _(e):
                self._safe(self._goto_preset, e.client)

            @center.on_click
            def _(e):
                self._safe(self._center, e.client)
            self.cam_preset.on_update(lambda e: e.client and self._safe(self.app.request_calibrated_render))
            self.calib_enable.on_update(lambda e: e.client and self._safe(self.app.request_calibrated_render))

    def selected_calibrated(self):
        for c in self.app.cfg.cameras:
            if c.name == self.cam_preset.value:
                return c
        return self.app.cfg.cameras[0] if self.app.cfg.cameras else None

    def _goto_preset(self, client):
        cam = self.selected_calibrated()
        if client is None:
            return
        if self.cam_preset.value == "free orbit" or cam is None:
            self._center(client)
            return
        T = cam.T_world_camera
        center = self.app.object_center()
        fwd = T[:3, 2]
        dist = max(0.05, float(np.dot(center - T[:3, 3], fwd)))
        with client.atomic():
            client.camera.fov = vertical_fov_from_K(cam.K, cam.height)
            client.camera.position = T[:3, 3]
            client.camera.look_at = T[:3, 3] + fwd * dist
            client.camera.up_direction = -T[:3, 1]
            client.camera.wxyz = quat_from_matrix(T[:3, :3])

    def _center(self, client):
        if client is None:
            return
        c = self.app.object_center()
        r = self.app.object_radius()
        with client.atomic():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.look_at = c
            client.camera.position = c + np.array([1.0, -1.4, 0.9]) / np.linalg.norm([1.0, -1.4, 0.9]) * 3.2 * r

    def update_camera_md(self):
        ev = self.app.render_cb.latest()
        if ev is None:
            return
        K = np.array(ev.K).reshape(3, 3)
        T = np.array(ev.c2w).reshape(4, 4)
        self.camera_md.content = (
            f"Last orbit frame (virtual K): {ev.width}×{ev.height}, fx=fy={K[0,0]:.1f}, "
            f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}; camera at ({T[0,3]:+.3f}, {T[1,3]:+.3f}, {T[2,3]:+.3f}) m")

    # ------------------------------------------------------------- render
    def _build_render(self):
        gui = self.server.gui
        app = self.app
        opt = app.render_cb.options
        with gui.add_folder("Render", expand_by_default=False):
            self.r_mode = gui.add_dropdown("Mode", RENDER_MODES, initial_value=opt.mode,
                                           hint="link_id adds L one-hot channels to the same gsplat call")
            self.r_dmin = gui.add_number("Depth min [m]", initial_value=opt.depth_min, step=0.05)
            self.r_dmax = gui.add_number("Depth max [m]", initial_value=opt.depth_max, step=0.05)
            self.r_sh = gui.add_slider("SH degree", min=0, max=app.renderer.scene.max_degree, step=1,
                                       initial_value=app.renderer.sh_degree)
            self.r_res = gui.add_slider("Max resolution [px]", min=128, max=1920, step=32,
                                        initial_value=app.viewer.render_tab_state.viewer_res)
            self.r_bg = gui.add_rgb("Background", tuple(int(255 * v) for v in app.cfg.render.background_rgb))
            self.r_occ = gui.add_checkbox("Occlude overlays by splats", opt.occlude_overlays)
            gui.add_markdown("Depth = expected optical-Z (m) from gsplat RGB+ED; background is NaN.")

        def changed(e):
            if e.client is None:
                return
            opt.mode = self.r_mode.value
            opt.depth_min = float(self.r_dmin.value)
            opt.depth_max = max(float(self.r_dmax.value), opt.depth_min + 1e-3)
            opt.sh_degree = int(self.r_sh.value)
            opt.occlude_overlays = bool(self.r_occ.value)
            app.viewer.render_tab_state.viewer_res = int(self.r_res.value)
            app.renderer.set_background(np.array(self.r_bg.value) / 255.0)
            app.rerender_all()
            app.request_calibrated_render()
        for h in (self.r_mode, self.r_dmin, self.r_dmax, self.r_sh, self.r_res, self.r_bg, self.r_occ):
            h.on_update(changed)

    # ------------------------------------------------------------- overlays
    def _build_overlays(self):
        gui = self.server.gui
        self.ov = {}
        with gui.add_folder("Overlays (debug, never replace splats)", expand_by_default=False):
            for k in OVERLAY_KINDS:
                cb = gui.add_checkbox(OVERLAY_LABELS[k], OVERLAY_DEFAULTS[k])
                self.ov[k] = cb
                self.app.overlays.set_visible(k, OVERLAY_DEFAULTS[k])
                cb.on_update(self._make_overlay_cb(k))

    def _make_overlay_cb(self, kind):
        def cb(e):
            self.app.overlays.set_visible(kind, bool(e.target.value))
        return cb

    # ------------------------------------------------------------- export
    def _build_export(self):
        gui = self.server.gui
        with gui.add_folder("Export", expand_by_default=False):
            b = gui.add_button("Save snapshot (RGB, depth, alpha, K, c2w, q, T_world_link)")
            t = gui.add_button("Save trajectory log")
            self.export_md = gui.add_markdown(f"Output: `{self.app.out_dir}`")

            @b.on_click
            def _(e):
                self._safe(self._export, e.client)

            @t.on_click
            def _(e):
                self._safe(self._export_traj)

    def _export(self, client):
        path = self.app.export_snapshot(client)
        self.export_md.content = f"Saved `{path}`"

    def _export_traj(self):
        path = self.app.export_trajectory()
        self.export_md.content = f"Saved `{path}`"

    # ------------------------------------------------------------- utils
    def _safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            self.app.report_error("gui", exc)
            return None
