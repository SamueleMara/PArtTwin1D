# Live interactive 3D GUIs

Two front-ends share the same `ArticulatedGaussianRenderer` and ManiSkill env:

| | `parttwin maniskill-gui` (desktop) | `parttwin gui` (browser) |
|---|---|---|
| Window | **ManiSkill's own viewer** (`env._viewer`, `sapien.utils.Viewer`) | viser web page (nerfview @ 4538024) |
| Gaussians ↔ camera | rendered from the SAPIEN viewer camera **in the same frame**, depth-composited with the SAPIEN scene → zero lag | streamed JPEG, ≈12 ms behind overlays |
| SAPIEN scene | collision proxies, joint axes, SAPIEN Articulation / Transform / Scene windows | — |
| Remote / headless | needs display + Vulkan | yes (SSH port forward) |
| Physics backend | `physx_cpu` | `physx_cpu`, `physx_cuda` |
| Calibrated cameras | pose + vertical FOV (exact K in exports) | exact-K panel + exports |

## ManiSkill viewer (`parttwin maniskill-gui`)

```bash
python -m parttwin1d.cli.maniskill_gui --config configs/examples/hinged_box.yaml   # or: make maniskill-gui
```

How it is wired (`src/parttwin1d/viewer/maniskill_viewer.py`, `sapien_shader.py`):

1. The env enables SAPIEN rendering and returns a viewer `CameraConfig` whose
   `ShaderConfig.shader_pack` is a **generated** pack: the installed SAPIEN 3.0.3
   `default` pack with only `composite0.frag` patched (two samplers
   `samplerGaussianColor/Depth` + a depth test against the gbuffer). Cached in
   `~/.cache/parttwin1d/sapien_shaders/`, regenerated if SAPIEN changes; the patch
   fails loudly if SAPIEN's shader anchors move.
2. The viewer is created exactly as ManiSkill's `render_human()` does
   (`sapien_utils.create_viewer(env._viewer_camera_config)` + `env._setup_viewer()`),
   then `GaussianLayerPlugin` is appended to `viewer.plugins`.
3. Every frame, in `before_render` (after SAPIEN applied mouse input and synced
   PhysX poses), the plugin converts the window camera (OpenGL model/projection →
   OpenCV c2w and K), renders gsplat from it with a black background
   (premultiplied colour), and uploads RGBA8 + float depth as textures; the
   composite pass blends `g.rgb + (1-g.a)·scene` wherever the Gaussians are nearer
   than SAPIEN geometry. Nothing is re-rendered while camera and state are unchanged.

Controls: SAPIEN viewer mouse (right-drag rotate, middle/shift pan, wheel zoom) and
its windows (the twin articulation is pre-selected, so SAPIEN's *Articulation*
window sliders also drive the joints and the splats follow). The
**PArtTwin1D Gaussian twin** window: show Gaussians, collision proxies (opaque;
they hide the splats inside them), render scale, SH degree, joint sliders
(kinematic set / dynamics PD target), sweep, kinematic/dynamics mode, run physics,
joint-0 force, calibrated-camera pose, free camera, export (composited viewer
frame + Gaussian layer + every calibrated camera with exact K + state).

Measured (RTX 4070 Laptop, 960×720, `tests/gui/test_maniskill_viewer.py`):
orbit **35 fps** with the layer re-rendered every frame (SAPIEN frame ≈12 ms,
gsplat ≈7 ms — CUDA and Vulkan time-slice the GPU — readback ≈5 ms, texture
upload ≈4 ms); gsplat depth vs SAPIEN's rasterized door proxy: median 8.5 mm
(proxy face is 10 mm in front of the splat plane). Use `--render-scale 0.5–0.75`
for higher frame rates on large real objects.

Limitations: `physx_cpu` only; SAPIEN's `RenderWindow.set_intrinsic_parameters`
produced an inconsistent projection in SAPIEN 3.0.3, so calibrated presets use
pose + vertical FOV (centred principal point) — exact K stays in exports/web GUI.

## Classical nerfstudio viewer

Not integrated: `pip install nerfstudio` (1.1.5) would downgrade **gsplat
1.5.3 → 1.4.0, numpy 2.4.6 → 1.26.4 and viser 1.1.1 → 0.2.7** (dry run), breaking
the pinned stack, and its viewer expects a nerfstudio `Pipeline`. The browser GUI
already runs on nerfview, the viewer core the nerfstudio project extracted for
gsplat (same viser frontend and camera conventions).

# Browser GUI (`parttwin gui`)

```bash
conda activate parttwin_sim && cd ~/PArtTwin1D && source scripts/activate_sim.sh
python -m parttwin1d.cli.gui --config configs/examples/hinged_box.yaml --host 127.0.0.1 --port 8080
# open http://127.0.0.1:8080   (Ctrl+C stops cleanly)
```

Stack: **viser 1.1.1** (browser 3D client + websocket server) + **nerfview @
4538024** (per-client render threads, camera → `render_fn`) + the same
`ArticulatedGaussianRenderer` used offline + the live ManiSkill articulation.
Every camera move or joint change triggers a **new gsplat rasterization** on
the server GPU; the browser receives the frame as the scene background (JPEG)
plus a depth map so overlays are composited against the splats.

## Panels

| Panel | Contents |
|---|---|
| Status | shared-q notice, clients, mode, snapshot revision, sim time, render resolution/latency, last orbit camera K and position, errors/warnings (render failures, CUDA OOM hint, empty alpha, command errors) |
| Joints | one slider per active URDF joint (units, URDF limits in hint), numeric box, q readout with zero/non-zero and limits, *Reset to q0* |
| Kinematic sweep playback | joint, Play / Pause / Step, Loop, Speed. Manual slider input pauses the sweep |
| Physics (dynamics preview) | Mode kinematic/dynamics, Run physics, input kind (PD drive target / generalized force), value, Apply, Step ×10. In dynamics mode sliders set the drive target |
| Camera | presets (free orbit + each calibrated camera), Go to preset pose, Center on object, *Calibrated view* (exact-K render of the selected real camera) |
| Render | mode rgb / alpha / depth (metric colorbar, min/max) / link_id, SH degree, max resolution, background colour, occlude overlays |
| Overlays | link frames, joint axes + hinge, labels with q, Gaussian AABBs, collision primitives, calibrated frusta, world axes, debug grid (debug only) |
| Export | Save snapshot → `outputs/<run_id>/snapshot_*/rev*/`: orbit view (virtual K) + every calibrated camera (exact K): RGB PNG, alpha PNG, depth NPY, NPZ, link-ID, metadata JSON (K, c2w, w2c, q, T_world_link) + `state.json`; Save trajectory log |

Mouse: left-drag orbit, right-drag (or shift) pan, wheel zoom.

## Camera semantics / limitations

* The orbit view uses a **virtual pinhole** (fx = fy from the browser FOV,
  centred principal point). nerfview cannot carry an off-centre calibrated
  principal point, so calibrated intrinsics are shown only in the *Calibrated
  view* panel (preview width 480 px, K rescaled exactly) and used in exports.
  *Go to preset pose* moves the orbit camera to the sensor pose with the sensor's
  vertical FOV, but it is still the virtual K.
* Frames are JPEG-compressed for transport (quality 70 static / 40 while
  moving); numeric parity with the offline renderer is tested before JPEG.
* The splats are rendered on the server, so during an orbit the image trails the
  browser-drawn overlays by one round trip. Measured camera update → frame sent:
  **≈12 ms median, 15 ms p90** at 960×853 (`tests/gui::test_camera_motion_latency`).
  To keep this low, colour conversion runs on the GPU (only uint8 is copied to the
  host) and the depth map used to occlude overlays (≈10 ms PNG encode per frame in
  viser) is sent only with nerfview's settled high-quality frame, ~0.2 s after
  motion stops. While the camera moves, overlays draw on top of the splats.
* Shared state: articulation q is shared by all browsers; cameras are per browser.
* The headless/software-WebGL warning shown by viser in VMs only affects the
  browser's overlay drawing, not the server-side gsplat render.
