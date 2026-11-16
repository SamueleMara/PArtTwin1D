# Live interactive 3D GUI

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
