"""Per-link 3D Gaussians inside ManiSkill's own SAPIEN viewer (``env.render_human()``).

How it works (one thread, one frame, one camera):

1. The env is created with SAPIEN rendering enabled and a viewer ``CameraConfig``
   whose shader pack is the generated compositing pack
   (:mod:`parttwin1d.viewer.sapien_shader`). ManiSkill's ``render_human()`` builds
   the standard ``sapien.utils.Viewer`` from it — the normal ManiSkill GUI.
2. :class:`GaussianLayerPlugin` (a SAPIEN viewer plugin) runs in
   ``before_render`` every frame — after the viewer has applied mouse input
   (ControlWindow.after_render of the previous frame) and synced PhysX poses
   (``window.update_render``). It reads the window camera (model + projection
   matrix), renders the Gaussians with gsplat from exactly that camera and the
   current PhysX link poses, and uploads premultiplied RGBA + optical-Z depth as
   the textures ``GaussianColor`` / ``GaussianDepth``.
3. SAPIEN rasterizes its scene (collision proxies, joint axes, frames) and the
   patched composite pass depth-composites the Gaussian layer into the same frame.

Result: the splats are locked to the SAPIEN camera with zero frame lag; there is
no network transport. Window camera conventions (verified, sub-pixel):
``T_world_camera(OpenCV) = M_model(OpenGL c2w) @ diag(1,-1,-1,1)``,
``fx = W/2·P00, fy = -H/2·P11, cx = W/2·(1-P02), cy = H/2·(1-P12)``.

Scope: ``physx_cpu`` only — the SAPIEN viewer's built-in ArticulationWindow and
the per-frame pose sync write/read CPU PhysX objects. (The web GUI supports both.)
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from ..geometry.cameras import look_at_c2w, vertical_fov_from_K
from ..rendering.cameras import RenderCamera
from ..rendering.observations import export_result, rgb_uint8

EMPTY_DEPTH = 1.0e9
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def window_render_camera(window, scale: float = 1.0) -> RenderCamera:
    """OpenCV RenderCamera equivalent to the SAPIEN viewer window camera."""
    try:
        W, H = (int(v) for v in window.get_picture_size("Color"))
    except IndexError:  # render targets do not exist before the first frame
        W, H = (int(v) for v in window.size)
    P = np.asarray(window.get_camera_projection_matrix(), np.float64)
    M = np.asarray(window.get_camera_model_matrix(), np.float64)
    K = np.array([[W / 2 * P[0, 0], 0.0, W / 2 * (1 - P[0, 2])],
                  [0.0, -H / 2 * P[1, 1], H / 2 * (1 - P[1, 2])],
                  [0.0, 0.0, 1.0]])
    cam = RenderCamera("sapien_viewer", W, H, K, M @ GL_TO_CV, "virtual")
    if scale != 1.0:
        cam = cam.resized(max(16, round(W * scale)), max(16, round(H * scale)))
    return cam


def sapien_pose_from_c2w(T_cv):
    """SAPIEN camera entity pose (x forward, y left, z up) from an OpenCV c2w."""
    import sapien

    from ..geometry.se3 import quat_from_matrix

    R = T_cv[:3, :3]
    R_s = np.stack([R[:, 2], -R[:, 0], -R[:, 1]], axis=1)
    return sapien.Pose(p=T_cv[:3, 3].tolist(), q=quat_from_matrix(R_s).tolist())


def _make_plugin_class():
    from sapien.utils.viewer.plugin import Plugin

    class GaussianLayerPlugin(Plugin):
        def __init__(self, app):
            self.app = app
            self._key = None
            self._size = None
            self._tex = None
            self.frames = 0
            self.last_ms = {"gsplat": 0.0, "upload": 0.0}

        # ---------------------------------------------------------- textures
        def _ensure_textures(self, w, h):
            import sapien

            if self._size == (w, h):
                return
            col = sapien.render.RenderTexture2D(np.zeros((h, w, 4), np.uint8), "r8g8b8a8unorm",
                                                filter_mode="linear", address_mode="edge")
            dep = sapien.render.RenderTexture2D(np.full((h, w, 1), EMPTY_DEPTH, np.float32), "r32sfloat",
                                                filter_mode="nearest", address_mode="edge")
            self.viewer.window.set_camera_texture("GaussianColor", col)
            self.viewer.window.set_camera_texture("GaussianDepth", dep)
            self._tex, self._size = (col, dep), (w, h)

        def init(self, viewer):
            super().init(viewer)
            self._ensure_textures(1, 1)

        def clear(self):
            self._size = None
            self._ensure_textures(1, 1)
            self._key = None

        # ---------------------------------------------------------- per frame
        def before_render(self):
            app = self.app
            try:
                app.enforce_proxy_visibility()
                if not app.show_gaussians:
                    if self._size != (1, 1):
                        self.clear()
                    return
                snap = app.current_snapshot()
                cam = window_render_camera(self.viewer.window, app.render_scale)
                key = (snap.revision, cam.width, cam.height, cam.K.tobytes(), cam.T_world_camera.tobytes(),
                       app.sh_degree)
                if key == self._key:
                    return  # nothing changed: keep the uploaded layer
                import torch

                t0 = time.perf_counter()
                r = app.renderer
                r.update_from_snapshot(snap)
                res = r.render(cam, background=(0.0, 0.0, 0.0), sh_degree=app.sh_degree)  # premultiplied
                rgba = torch.cat([res.rgb, res.alpha[..., None]], -1)
                rgba8 = (rgba.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
                depth = torch.nan_to_num(res.depth, nan=EMPTY_DEPTH)[..., None].cpu().numpy()
                t1 = time.perf_counter()
                self._ensure_textures(cam.width, cam.height)
                self._tex[0].upload(rgba8)
                self._tex[1].upload(depth)
                t2 = time.perf_counter()
                self.last_ms = {"gsplat": (t1 - t0) * 1e3, "upload": (t2 - t1) * 1e3}
                self.last_camera = cam
                self.last_result = res
                self._key = key
                self.frames += 1
            except Exception as exc:  # never kill the viewer loop
                app.report_error("gaussian layer", exc)

        def get_ui_windows(self):
            try:
                return self.app.build_panel()
            except Exception as exc:
                self.app.report_error("panel", exc)
                return []

    return GaussianLayerPlugin


class ManiSkillGaussianViewer:
    """ManiSkill ``render_human`` viewer with the Gaussian layer and a twin panel."""

    def __init__(self, cfg, *, width: int = 1280, height: int = 800, render_scale: float = 1.0,
                 collision_proxies: bool = True, out_dir="outputs", device: str = "cuda", sim_backend: str = "physx_cpu"):
        from ..rendering.gsplat_backend import ArticulatedGaussianRenderer
        from ..sim.articulation import ArticulationDriver
        from ..sim.env import create_env
        from .sapien_shader import build_shader_pack

        if sim_backend != "physx_cpu":
            raise ValueError("the ManiSkill/SAPIEN viewer mode supports sim_backend='physx_cpu' only "
                             "(use `parttwin gui` for physx_cuda)")
        self.cfg = cfg
        self.out_dir = Path(out_dir).resolve() / time.strftime(f"maniskill_gui_{cfg.object_name}_%Y%m%d_%H%M%S")
        self.shader_pack = build_shader_pack()
        self.renderer = ArticulatedGaussianRenderer.from_config(cfg, device=device)
        # initial viewer camera: same framing as the web GUI
        self.env = create_env(cfg, sim_backend, render_backend="gpu", render_mode="human",
                              collision_proxies=collision_proxies,
                              viewer_options={"shader_pack": self.shader_pack, "width": width, "height": height,
                                              "fov": 0.9})
        self.env.reset(seed=0)
        self.driver = ArticulationDriver(self.env, cfg)
        self.renderer.bind_articulation(self.env.twin)
        self._snap = self.driver.snapshot("reset")
        self._snap_sig = None
        self.renderer.update_from_snapshot(self._snap)

        # UI state (bound to ImGui widgets)
        self.show_gaussians = True
        self._show_proxies = False
        self.collision_proxies = collision_proxies
        self.render_scale = float(render_scale)
        self.sh_degree = self.renderer.sh_degree
        self.mode_index = 0  # 0 kinematic, 1 dynamics
        self.run_physics = False
        self.playing = False
        self.play_speed = 1.0
        self._phase = 0.0
        self.force_value = 0.0
        self.camera_index = 0
        self.status_lines: list[str] = []
        self.errors: list[str] = []
        self._last_t = time.perf_counter()
        self._fps = 0.0

        # Exactly ManiSkill's render_human() viewer creation (sapien_env.py), split so
        # the Gaussian textures are bound before the first frame is drawn.
        from mani_skill.utils import sapien_utils

        self.env._viewer = sapien_utils.create_viewer(self.env._viewer_camera_config)
        self.env._setup_viewer()
        self.viewer = self.env._viewer
        self.plugin = _make_plugin_class()(self)
        self.plugin.init(self.viewer)
        self.viewer.plugins.append(self.plugin)
        self._set_proxy_visibility(False)
        self._frame_object()
        # Pre-select the articulation so SAPIEN's ArticulationWindow shows its joints.
        # Deferred to after the first frame: SAPIEN plugins build their UI lazily.
        self._pending_select = self.env.twin.links_map[self.driver.urdf.root_links[0]]._objs[0].entity

    # ---------------------------------------------------------------- helpers
    @property
    def window(self):
        return self.viewer.window

    def report_error(self, source, exc):
        msg = f"{source}: {type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else f"{source}: {exc}"
        print(f"[parttwin maniskill-gui] ERROR {msg}", flush=True)
        if isinstance(exc, BaseException):
            traceback.print_exc()
        self.errors = ([msg] + self.errors)[:4]

    def object_center_radius(self):
        m = self.renderer.world_means()
        lo, hi = m.min(0).values, m.max(0).values
        return ((lo + hi) / 2).cpu().numpy().astype(np.float64), float(((hi - lo) / 2).norm().item()) or 0.5

    def _frame_object(self):
        c, r = self.object_center_radius()
        d = np.array([1.0, -1.4, 0.9]) / np.linalg.norm([1.0, -1.4, 0.9])
        self.viewer.set_camera_pose(sapien_pose_from_c2w(look_at_c2w(c + d * 3.2 * r, c)))

    def _set_proxy_visibility(self, visible: bool):
        self._show_proxies = bool(visible)
        self.enforce_proxy_visibility()

    def enforce_proxy_visibility(self):
        """Called every frame: SAPIEN's Viewer.select_entity() overrides the selected
        entity's visibility (highlight 0.5, previous one reset to 1), which would
        re-show hidden proxies and occlude the splats inside them."""
        import sapien

        if not self.collision_proxies:
            return
        want = 1.0 if self._show_proxies else 0.0
        for link in self.env.twin.links:
            for obj in link._objs:
                rb = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if rb is not None and rb.visibility != want:
                    rb.visibility = want

    @property
    def show_proxies(self):
        return self._show_proxies

    @show_proxies.setter
    def show_proxies(self, v):
        self._set_proxy_visibility(bool(v))

    def current_snapshot(self):
        """Fresh snapshot when q / link poses changed (also via SAPIEN's own joint sliders)."""
        art = self.env.twin
        q = art.get_qpos()[0].numpy().copy()
        sig = q.tobytes() + b"".join(art.links_map[n].pose.raw_pose[0].numpy().tobytes()
                                     for n in self.driver.bound_links)
        if sig != self._snap_sig:
            mode = "dynamics" if self.mode_index == 1 else "kinematic"
            self._snap = self.driver.snapshot(mode)
            self._snap_sig = sig
        return self._snap

    # ---------------------------------------------------------------- actions
    def set_joint(self, name, value):
        try:
            if self.mode_index == 1:
                self._drive_target = {**getattr(self, "_drive_target", {}), name: float(value)}
            else:
                self.playing = False
                self.driver.set_joint_kinematic(name, float(value))
        except Exception as exc:
            self.report_error("joint", exc)

    def reset(self):
        self.playing = False
        self.run_physics = False
        self._drive_target = {}
        self.driver.reset_to_q0()

    def snap_to_camera(self):
        if not self.cfg.cameras:
            return
        spec = self.cfg.cameras[self.camera_index % len(self.cfg.cameras)]
        self.window.set_camera_parameters(0.01, 100.0, vertical_fov_from_K(spec.K, spec.height))
        self.viewer.set_camera_pose(sapien_pose_from_c2w(spec.T_world_camera))

    def free_camera(self):
        self.window.set_camera_parameters(0.01, 100.0, 0.9)
        self._frame_object()

    def export(self):
        out = self.out_dir / time.strftime("snapshot_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        pic = np.asarray(self.window.get_picture("Color"))[..., :3]
        Image.fromarray(rgb_uint8(pic)).save(out / "viewer_composited.png")
        snap = self.current_snapshot()
        self.renderer.update_from_snapshot(snap)
        cam = window_render_camera(self.window)
        export_result(self.renderer.render(cam, segmentation=True), out, "viewer_gaussians",
                      {"camera_kind": "SAPIEN viewer camera (FOV-based pinhole)"})
        for spec in self.cfg.cameras:
            export_result(self.renderer.render(RenderCamera.from_spec(spec), segmentation=True), out, spec.name,
                          {"camera_kind": "calibrated exact K"})
        (out / "state.json").write_text(json.dumps({"snapshot": snap.to_json()}, indent=2))
        self.status_lines = [f"saved {out}"]
        return out

    # ---------------------------------------------------------------- loop
    def step(self):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        try:
            if self.playing and self.mode_index == 0:
                j = next(j for j in self.driver.joints if j.bounded)
                self._phase = (self._phase + dt * self.play_speed / 4.0) % 1.0
                tri = 2 * self._phase if self._phase < 0.5 else 2 * (1 - self._phase)
                self.driver.set_joint_kinematic(j.name, j.lower + (j.upper - j.lower) * tri)
            if self.run_physics and self.mode_index == 1:
                n = max(1, round(dt / self.env.control_timestep)) if dt < 0.25 else 1
                tgt = getattr(self, "_drive_target", {})
                if self.force_value:
                    f = np.zeros(self.driver.dof)
                    f[0] = self.force_value
                    self.driver.step_dynamics(n, joint_force=f)
                else:
                    q = self.driver.get_q()
                    for name, v in tgt.items():
                        q[self.driver.joint_index[name]] = v
                    self.driver.step_dynamics(n, drive_target=q)
        except Exception as exc:
            self.playing = self.run_physics = False
            self.report_error("update", exc)

    def run(self, max_frames: int | None = None, screenshot: str | None = None):
        n = 0
        try:
            while not self.viewer.closed:
                self.step()
                self.env.render_human()
                n += 1
                if self._pending_select is not None:
                    entity, self._pending_select = self._pending_select, None
                    try:
                        self.viewer.select_entity(entity)
                    except Exception as exc:
                        self.report_error("select", exc)
                if max_frames is not None and n >= max_frames:
                    if screenshot:
                        from PIL import Image

                        pic = np.asarray(self.window.get_picture("Color"))[..., :3]
                        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rgb_uint8(pic)).save(screenshot)
                    break
        finally:
            self.close()

    def close(self):
        try:
            if not self.viewer.closed:
                self.viewer.close()
        except Exception:
            pass
        self.env.close()

    # ---------------------------------------------------------------- ImGui panel
    def build_panel(self):
        import sapien.internal_renderer as R

        snap = self._snap
        ms = self.plugin.last_ms
        cam = getattr(self.plugin, "last_camera", None)
        info = [
            f"{self.cfg.object_name}: {self.renderer.gaussian_count} Gaussians (gsplat 1.5.3, SH {self.sh_degree})",
            f"viewer {self._fps:5.1f} fps | gsplat {ms['gsplat']:.1f} ms + upload {ms['upload']:.1f} ms"
            + (f" @ {cam.width}x{cam.height}" if cam is not None else ""),
            f"mode {'dynamics' if self.mode_index else 'kinematic'} | sim t {snap.sim_time:.2f} s | rev {snap.revision}",
            "Gaussians are depth-composited with the SAPIEN scene in the same frame.",
        ] + self.status_lines + [f"ERROR {e}" for e in self.errors]
        win = R.UIWindow().Label("PArtTwin1D Gaussian twin").Pos(10, 440).Size(430, 560)
        for line in info:
            win.append(R.UIDisplayText().Text(line))
        win.append(
            R.UICheckbox().Label("Show Gaussians (gsplat layer)").Bind(self, "show_gaussians"),
            R.UICheckbox().Label("Show collision proxies (occlude splats)").Bind(self, "show_proxies"),
            R.UISliderFloat().Label("Gaussian render scale").Min(0.25).Max(1.0).Bind(self, "render_scale"),
            R.UIInputInt().Label("SH degree").Bind(self, "sh_degree"),
        )
        joints = R.UISection().Label("Joints (kinematic: set q | dynamics: PD target)").Expanded(True)
        for j in self.driver.joints:
            lo, hi = (j.lower, j.upper) if j.bounded else (-np.pi, np.pi)
            q = float(snap.q[j.index])
            joints.append(
                R.UISliderFloat().Label(f"{j.name} [{j.unit}]").Id(f"pt_joint_{j.index}").Min(lo).Max(hi).Value(q)
                .Callback((lambda name: lambda p: self.set_joint(name, p.value))(j.name)))
        joints.append(R.UIButton().Label("Reset to q0").Callback(lambda _: self.reset()))
        win.append(joints)
        pb = R.UISection().Label("Kinematic sweep / physics").Expanded(True)
        pb.append(
            R.UICheckbox().Label("Play sweep (kinematic)").Bind(self, "playing"),
            R.UISliderFloat().Label("Speed").Min(0.1).Max(4.0).Bind(self, "play_speed"),
            R.UIOptions().Style("select").Label("Mode").Items(["kinematic", "dynamics"]).BindIndex(self, "mode_index"),
            R.UICheckbox().Label("Run physics (dynamics)").Bind(self, "run_physics"),
            R.UISliderFloat().Label("Joint 0 force/torque").Min(-5.0).Max(5.0).Bind(self, "force_value"),
        )
        win.append(pb)
        cams = R.UISection().Label("Cameras / export").Expanded(True)
        if self.cfg.cameras:
            cams.append(
                R.UIOptions().Style("select").Label("Calibrated").Items([c.name for c in self.cfg.cameras])
                .BindIndex(self, "camera_index"),
                R.UIButton().Label("Go to calibrated pose (pose + vertical FOV)").Callback(lambda _: self.snap_to_camera()),
            )
        cams.append(R.UIButton().Label("Free camera / center object").Callback(lambda _: self.free_camera()),
                    R.UIButton().Label("Export snapshot").Callback(lambda _: self.export()))
        win.append(cams)
        return [win]
