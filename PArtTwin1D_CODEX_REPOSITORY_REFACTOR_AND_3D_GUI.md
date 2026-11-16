# CODEX IMPLEMENTATION DIRECTIVE — PArtTwin1D
## Production repository architecture, pinned upstream submodules, ManiSkill articulated 3DGS, and interactive 3D GUI

**Priority:** implement and verify, not propose another smoke package.  
**Existing working directory:** `/home/samuelemara/PArtTwin1D` (`~/PArtTwin1D`).  
**Existing Conda environment:** `parttwin_sim` (Python 3.11.18).  
**Installed baseline reported by the operator:** PyTorch `2.11.0+cu128`, CUDA toolkit `12.8`, ManiSkill `3.0.1`, SAPIEN `3.0.3`, gsplat `1.5.3`, plyfile `1.1.5`, scipy `1.17.1`, RTX 4070 Laptop (SM 8.9).  
**Do not** recreate the Conda environment, clone `PArtTwin1D` again, use a temporary `parttwin_smoke` package, or install a new ML/physics stack without a demonstrated compatibility reason.

> **Definition of done:** From the existing repository root, a developer can load a user-supplied articulated URDF with per-link trained Gaussian PLYs, open a **live interactive 3D browser GUI**, orbit/pan/zoom around the real Gaussian-rendered object, move each joint with sliders or physics stepping, see correct live link-pose updates and overlays, save RGB/depth/alpha and trajectories, and run automated geometry, integration, and UI checks. Upstream source dependencies are reproducibly referenced by committed Git submodules. Tests must be observed actually passing on the operator's GPU; no mocked tests count as proof of graphics/physics integration.

---

## 0. MANDATORY FIRST ACTION: REPOSITORY AUDIT, NOT NEW CODE

Inspect the **actual local checkout**; it may contain files that are not on the remote `main` branch. Do not equate GitHub's remote README-only view with the local working tree.

```bash
cd ~/PArtTwin1D
pwd
git status --short
git branch --show-current
git remote -v
git ls-files | sort
find . -maxdepth 3 -type f -not -path './.git/*' | sort | head -250
find . -maxdepth 5 -name '*parttwin_smoke*' -o -name 'test_sh_directions.py' -o -name 'test_transforms.py'
python -c 'import sys,importlib.util; print(sys.executable); print(importlib.util.find_spec("parttwin_smoke")); print(importlib.util.find_spec("parttwin1d"))'
python -m pip check
```

1. Record `git status` and do **not discard uncommitted work**. Work on a descriptive branch such as `feature/maniskill-3dgs-gui` (unless the operator has already selected one). Create a baseline commit only if changes are understood and approved by repository convention; do not silently commit unrelated work.
2. Audit the existing `parttwin1d/sim/{assets,env,geometry,renderer}.py`, `scripts/{activate_sim,check_sim,inspect_asset,create_demo_assets,render_articulated}`, configs, and tests **as currently present**. If moved into `src/`, preserve behavior and update all import locations.
3. Identify duplicated/dead `parttwin_maniskill_gs_smoke`, `parttwin_maniskill_gs_v2`, `parttwin_maniskill_gs_v3`, editable imports from `~/.local/share/Trash/files/...`, and stale root tests. **Do not delete files without checking whether they are tracked or contain user changes.** Move obsolete tracked sources with `git mv` to `legacy/experiments/` and quarantine untracked generated files with a manifest; exclude legacy from test discovery. The production import namespace must be only `parttwin1d`.
4. Fix the currently observed **real** failure: root `tests/test_sh_directions.py` and `tests/test_transforms.py` still import `parttwin_smoke`. They must be migrated to `parttwin1d` tests if meaningful, otherwise archived outside pytest discovery. Simply running `pytest tests/unit` and leaving broken root tests behind does not count as cleanup: `python -m pytest` from repository root must work.
5. Identify the point where the existing script logs `Loaded 1480 Gaussians across 2 URDF links` and then blocks/compiles/fails. Distinguish actual gsplat JIT build time from a deadlock or API/graphics error. Capture full log, return code, wall time, and extension path. Never claim success from the print preceding a rasterization call.
6. Read the **installed** ManiSkill 3.0.1 and gsplat 1.5.3 source, and inspect exact runtime method signatures with `inspect.signature` or source files before changing integration code. Pin the sources before assuming current `main` APIs match releases.

### Previous logs and pitfalls to preserve as regressions

- User environment and imports worked: torch CUDA, ManiSkill, SAPIEN, gsplat, plyfile, scipy.
- Root `pytest` **failed during collection** because old tests imported `parttwin_smoke`; this is a namespace/repository hygiene fault, not evidence the current new modules are broken.
- Earlier robot-free `BaseEnv(robot_uids="none")` failed in ManiSkill's base `get_state_dict()` because `self.agent` was `None`. The existing production `ArticulatedTwinEnv` overrides `get_state_dict()` and `_get_obs_agent()`; preserve and cover that behavior.
- ROS Jazzy leaked pytest plugins from `/opt/ros/jazzy/lib/python3.12/...` into Conda Python 3.11. Test isolation must be repository-controlled, not require ROS dependencies: e.g., set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` inside `scripts/test.sh` and document `PYTHONPATH` isolation.
- `gsplat 1.5.3` with PyTorch 2.11 may hit a caught `_jit_compile()` signature `TypeError` before fallback; inspect the **terminal** compiler result. Reuse persistent CUDA extension cache, do not invoke repeated compiles by changing cache directories or compiler flags arbitrarily.
- One of the previous editable installs imported a package from Trash. Remove stale distribution links **only after confirming their location**; provide a runtime provenance report of `module.__file__` and `importlib.metadata.version()`.

---

## 1. REQUIRED PRODUCTION FOLDER STRUCTURE

Refactor into a single coherent tree. Adapt names only for clear reasons and document deviations.

```text
PArtTwin1D/
├── .gitmodules
├── .gitignore
├── README.md
├── pyproject.toml                  # Project metadata + development/test entrypoints
├── Makefile                        # setup, doctor, test, test-gpu, gui, render, lint
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   ├── coordinate_frames.md
│   ├── urdf_and_gaussians.md
│   ├── gui.md
│   ├── testing.md
│   └── third_party.md             # URL, pinned SHA, license, purpose, runtime import
├── third_party/                   # ONLY pinned Git submodules (no random clones)
│   ├── ManiSkill/
│   ├── gsplat/
│   └── nerfview/                 # Actual reusable interactive neural-render viewer
├── src/
│   └── parttwin1d/
│       ├── __init__.py
│       ├── config/
│       │   ├── schema.py
│       │   ├── loader.py
│       │   └── validate.py
│       ├── assets/
│       │   ├── gaussian_ply.py
│       │   ├── urdf.py
│       │   └── frame_alignment.py
│       ├── geometry/
│       │   ├── se3.py
│       │   └── cameras.py
│       ├── sim/
│       │   ├── env.py             # ManiSkill scene, PhysX articulation, no fake robot
│       │   ├── articulation.py    # Kinematic/physics commands, joint discovery
│       │   └── state.py           # q, dq, link world poses, timestamps
│       ├── rendering/
│       │   ├── gaussians.py       # Read-only canonical per-link splats
│       │   ├── link_adapter.py    # Exact T_world_link * T_link_ply transforms
│       │   ├── gsplat_backend.py  # One backend; no custom CUDA rasterizer
│       │   ├── cameras.py         # Intrinsics and conventions
│       │   └── observations.py    # GPU RGB/depth/alpha interface for future PPO
│       ├── viewer/
│       │   ├── server.py          # viser + nerfview host
│       │   ├── render_callback.py # Camera state -> gsplat render
│       │   ├── controls.py        # Joint sliders, play/pause, render options
│       │   ├── overlays.py        # Joint axis/link frames/camera frusta
│       │   └── coordination.py    # Thread-safe sim/render state scheduling
│       ├── evaluation/
│       │   ├── consistency.py     # Geometry/joint/render invariants
│       │   └── profiling.py
│       └── cli/
│           ├── doctor.py
│           ├── inspect.py
│           ├── render.py
│           └── gui.py
├── configs/
│   ├── examples/hinged_box.yaml
│   ├── real_object.template.yaml
│   └── gui.defaults.yaml
├── assets/
│   ├── README.md
│   └── .gitkeep                    # Real large captures/PLY typically gitignored
├── examples/
│   ├── hinged_box/                # Deterministic tiny tracked fixture
│   └── demo_articulated.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gui/
├── scripts/
│   ├── bootstrap_submodules.sh
│   ├── check_environment.sh
│   ├── test.sh
│   ├── test_gpu.sh
│   └── run_gui.sh
├── requirements/
│   ├── runtime.in
│   ├── gui.in
│   ├── dev.in
│   └── environment.lock.md        # Exact verified versions/SHAs, reproducible install
├── outputs/                        # gitignored; captures/logs/profiles
└── legacy/experiments/             # Archived old smoke/v2/v3, NOT on PYTHONPATH
```

Use either a proper `src/parttwin1d` editable project install **for the actual project**, or a documented repo-root launcher, consistently. **Never** publish/install `parttwin-maniskill-gs-smoke` again. Prefer `python -m parttwin1d.cli.gui`, `python -m parttwin1d.cli.render`, and real `parttwin` console entrypoints defined in `pyproject.toml`. Do not create parallel implementations of SE(3), GS loading, or SAPIEN state access.

The external GPU drivers, SAPIEN binary wheel, CUDA toolkit, and Git-submodule sources cannot be made literally independent of the host machine; make their versions/retrieval **reproducible**, not fictitiously vendorized.

---

## 2. REPRODUCIBLE UPSTREAM SOURCES — GIT SUBMODULES

The user expressly wants actual imported/cloned upstream projects organized as Git submodules, **not unversioned local copies**. Implement with `git submodule`, not `git clone` scattered around the root. Pin exact commit SHAs in the superproject and document licenses.

| Submodule | Required | Pinning / purpose |
|---|---|---|
| `third_party/ManiSkill` | Yes | `https://github.com/mani-skill/ManiSkill.git`, inspect and pin release tag `v3.0.1`, matching installed `mani_skill==3.0.1` |
| `third_party/gsplat` | Yes | `https://github.com/nerfstudio-project/gsplat.git`, pin release tag `v1.5.3`, matching installed gsplat `1.5.3` |
| `third_party/nerfview` | Yes | `https://github.com/nerfstudio-project/nerfview.git`, select and record **one verified compatible commit SHA**; use its viewer callback and `rerender()` implementation; verify API with installed packages |
| `viser` | Python runtime dependency | Reuse the viewer's Viser frontend; pin compatible installed distribution in lockfile. Add as an additional submodule only if you genuinely modify/debug its source. |
| `ManiSkill-GS`, `GSWorld`, `GaussianRenderer` | **Reference only, not required** | If Codex needs one, add to `third_party/references/` as an explicitly optional, pinned submodule. Do **not** install their old pinned dependency stacks over the known-working environment. |
| `SAPIEN 3.0.3` | Binary dependency | Pin the pip/Conda distribution. Do **not** invent a matching Python-source submodule that is not used. |

Bootstrap safely, idempotently (first check existing `.gitmodules` and paths):

```bash
cd ~/PArtTwin1D
# Only if corresponding submodule not already recorded:
git submodule add https://github.com/mani-skill/ManiSkill.git third_party/ManiSkill
git -C third_party/ManiSkill checkout v3.0.1

git submodule add https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
git -C third_party/gsplat checkout v1.5.3

git submodule add https://github.com/nerfstudio-project/nerfview.git third_party/nerfview
# Checkout an inspected, compatible full SHA here; NOT blindly origin/main.
# git -C third_party/nerfview checkout <INSPECTED_SHA>

git submodule update --init --recursive

git submodule status
```

**Runtime source provenance must be explicit:** the target **source-controlled development mode** imports `mani_skill`, `gsplat`, and `nerfview` from their pinned `third_party/` submodules (for example, editable upstream installs with `python -m pip install --no-deps -e third_party/ManiSkill` after checking the exact upstream packaging and compatibility). The pre-existing pinned wheels may be kept temporarily as a **verified fallback** if forcing editable gsplat would invalidate CUDA compilation; this mode must be labelled `wheel-fallback`, and Codex must diagnose the difference before claiming the repository is self-contained in source. Document ONE selected runtime mode, verify every `module.__file__`, and never silently mix APIs from `main` with installed pinned wheels. Do not reinstall Torch or rebuild CUDA unnecessarily. The project itself should work as a normal `src/` package.

Implement `parttwin doctor --provenance` to output versions, absolute import paths, submodule SHAs, `CUDA_HOME`, `nvcc --version`, torch CUDA, device capability, and a version compatibility verdict. Fail loudly on incompatible code/runtime tags. Provide reproducible `git clone --recurse-submodules ...` and `git submodule update --init --recursive` instructions in `docs/installation.md`, but **do not clone another PArtTwin1D in the current task**.

---

## 3. FULL MANISKILL / PHYSX INTEGRATION (NO RECONSTRUCTED ROBOT)

### Scope

- Input: user URDF containing fixed and movable links, estimated joint type/axis/limits, collision geometries and inertial/dynamics fields where known; and canonical Gaussian PLYs per real object link.
- For this task: **one articulated object, no robot**, no modeled background 3DGS, no PPO. Empty background or colored backdrop is fine.
- The engine must load the URDF through ManiSkill/SAPIEN's actual URDF loader and derive **actual PhysX articulation link poses**; no homegrown forward-kinematics engine posing as SAPIEN.
- Use `render_backend="none"` for physics-only headless simulation if the external gsplat viewer renders all appearance. A separate `debug-physx` mode may enable SAPIEN's standard mesh viewer; it is **not** a Gaussian viewer and must not be advertised as such.

### Compatibility issue to handle explicitly

In ManiSkill 3.0.1, `BaseEnv`'s `get_state_dict()` and `_get_obs_agent()` assume an agent. For `robot_uids="none"`, implement and test robot-free overrides. Also inspect `evaluate`, `reset`, `get_info`, and initial observation space construction so no accidental agent dereference remains. Do not solve this by instantiating a dummy Franka.

### Joint synchronization

- Discover active joints by **name**, reading `Articulation.active_joints`, `get_active_joints()`, `get_qlimits()` for actual order and range; never equate all URDF joints with active degrees of freedom (fixed/mimic joints differ).
- `Kinematic preview` mode: set q with official ManiSkill/SAPIEN APIs, zero qvel if needed, update GPU kinematics/fetch where applicable; never call it a physics-forward simulated trajectory.
- `Dynamics preview` mode: step PhysX via official scene/control APIs with joint drives or generalized forces, validate contact/friction/inertia, and clearly label the external input; this must be separate from kinematic preview.
- Provide `get_state_snapshot()` returning immutable: q, qdot, `T_world_link` for **every bound Gaussian link**, simulation timestamp, environment index and joint status. Renderer consumes this snapshot without mutating it.
- Fixed root initial pose must be explicitly configured and applied before PhysX setup, not implicit identity with a warning.
- **No GPU-to-CPU conversion inside a future vectorized policy observation hot path.** First version can use NumPy link transforms for single scene/UI, but design a batched Torch tensor path and clearly document the performance boundary.

---

## 4. MATHEMATICAL FRAME CONTRACT (HARD REQUIREMENT)

Define a single matrix notation everywhere:

\[
T_{A\leftarrow B}\begin{bmatrix}x_B\\1\end{bmatrix}=\begin{bmatrix}x_A\\1\end{bmatrix},
\qquad
T_{W\leftarrow P_i}(q)=T_{W\leftarrow L_i}^{\mathrm{PhysX}}(q)\,T_{L_i\leftarrow P_i}.
\]

- `W`: simulator world, `L_i`: URDF link frame, `P_i`: canonical PLY frame.
- For Gaussian mean: `mu_W = R_WP mu_P + t_WP`.
- For covariance: `Sigma_W = R_WP Sigma_P R_WP^T`.
- For Gaussian orientation quaternion, choose and document scalar-first convention and multiplication order consistent with gsplat; test against explicit `R_WP R_gaussian` matrix composition.
- For view-dependent SH: preserve **original canonical Gaussian-local view directions** or rotate SH coefficients consistently; do not rotate geometry while leaving the SH basis orientation ambiguous. Test degree 0, 1 and 3 with moving hinges.
- Never automatically translate splats to centroid, rescale, swap axes, or assume world PLY frame is equivalent to URDF link frame. Reject missing/invalid transforms.
- Alignment from original captured world and reference q0 must be proven algebraically (including `T_simWorld_captureWorld`, if applicable); do not infer `T_link_ply=I` by visual appearance.
- Units: URDF meters, revolute radians, prismatic meters; if source reconstruction uses different scale it must be corrected explicitly and validated against collision geometry.
- Validate 4x4 homogeneous transforms, right-handed `det(R)=+1`, finite values, camera K shapes, URDF link existence, q limits, PLY Gaussian attribute presence.

**Camera contract:** Store full calibrated `K=[[fx,s,cx],[0,fy,cy],[0,0,1]]` (where renderer supports nonzero skew explicitly; otherwise reject skew and explain), image width/height, and **OpenCV optical** `T_world_camera` (+X right, +Y down, +Z forward). gsplat takes `viewmats = inverse(T_world_camera)` and `[C,3,3]` Ks. Do not confuse c2w and w2c. Calibrated real cameras and user-controlled GUI cameras are two distinct camera sources using the same renderer.

**Important viewer convention:** Current Viser documentation states that Viser cameras already use **OpenCV/COLMAP (+Z forward, +Y down)**. Do **not** apply an extra OpenGL-to-OpenCV 180-degree conversion unless a specifically inspected API really uses OpenGL. Validate on a known colored coordinate-frame fixture.

---

## 5. RENDERER BACKEND: EXISTING GSPLAT, SINGLE IMPLEMENTATION

Reuse the installed gsplat CUDA kernels, version `1.5.3`. **Do not write a new CUDA Gaussian rasterizer, train new Gaussians, or substitute a point cloud or mesh as the main image.**

Implement:

```python
renderer = ArticulatedGaussianRenderer(asset_manifest, device="cuda")
renderer.bind_articulation(articulation)  # validate exact link mappings
renderer.update_from_snapshot(snapshot)
result = renderer.render(camera)          # GPU float RGB[H,W,3], alpha[H,W], depth[H,W]
```

Technical requirements:

1. Decode Graphdeco-style 3DGS PLY accurately: means, log-scale, opacity logits, scalar-first quaternion, `f_dc_*`, channel-major `f_rest_*` SH. Distinguish other variants by schema or provide explicit import adapters (do not silently parse XYZ/RGB as splats). Preserve all coefficients, no automatic subsampling.
2. Keep each part's **canonical arrays immutable**. Transform means/orientations from the current snapshot each frame rather than repeatedly modifying the previous frame's arrays (otherwise link motion drifts numerically).
3. Composite all links in **one gsplat call** per camera, with global depth-aware occlusion and stable segmentation labels. Do **not** alpha-paste one RGBA link image on top of another in arbitrary link order.
4. SH either be evaluated by gsplat with correctly rotated directions or in canonical part frame before passing RGB to gsplat; inspect exact version `1.5.3` shapes/signs. Regression test against reference 3DGS rotated geometries.
5. Render RGB `[0,1]` float tensor, alpha `[0,1]` float, and expected optical Z depth in **meters**; clarify depth semantics on transparent/background pixels, use NaN/validity mask. Add optional link-ID passes for visualization/debug, with performance documented.
6. Full `fx,fy,cx,cy`; no implicit `fx=fy`, `cx=W/2`, `cy=H/2` in real calibrated camera mode.
7. Persistent GPU buffers; load PLY to GPU **once**, no unnecessary `.cpu().numpy()`, `torch.cuda.empty_cache()` or compiler invocation per frame. Cache extension using the existing `$CONDA_PREFIX` toolchain. Handle `CUDA_HOME` and SM 8.9; keep errors/logs visible.
8. Provide a separate CPU-export method for PNG/GIF/NPZ and JSON metadata. Inference only; no grads unless later implementing a training module.
9. `RenderResult` typed dataclass includes camera intrinsics/extrinsics, q, timestamp, RGB, alpha, depth and optional segmentation; validate shapes, dtype and device.
10. At minimum, support one environment and one calibrated camera; architect camera batching and multi-environment rendering without claiming GPU PPO throughput has already been achieved.

---

## 6. REQUIRED **LIVE INTERACTIVE 3D GUI** — NOT AN ANIMATED GIF

### Chosen reuse path

Use **`nerfview` + `viser`**, wired to **the same** `ArticulatedGaussianRenderer` and live PhysX articulation. This reuses an existing interactive neural-rendering viewer instead of implementing a second rendering/UI engine. Viser runs a 3D web interface; nerfview already supplies render callback plumbing and user camera updates. Inspect the pinned `third_party/nerfview` source (`Viewer`, `CameraState`, its `render_fn`, `Viewer.rerender()`) and installed versions before coding. Relevant official upstream examples: `gsplat/examples/simple_viewer.py`, `gsplat/examples/gsplat_viewer.py`.

**Do not accept a GUI that only loads a saved `sweep.gif`/PNG or displays a static PLY point cloud.** The GUI must call live gsplat render on camera movement and joint updates and show **actual Gaussian RGB**.

### Required interaction and display

- Browser-accessible via `http://127.0.0.1:8080` by default, with explicit `--host` and `--port`, and docs for SSH forwarding. Avoid binding public interfaces by default.
- Orbit, pan, zoom and center-on-object; free camera pose and projection updated in gsplat. Show scene axes and camera view frusta as optional geometric overlays.
- A scrollable joint list discovered from actual URDF active joints, with **one slider per joint**, numerical q value, units (rad/m), URDF limits, reset-to-q0, and zero/nonzero joint status.
- Play, pause, step, loop, playback-speed control for a kinematic joint sweep; explicit button/switch to run **physical** stepping (drive/forces) if implemented. Playback and manual input must not race.
- Camera presets: each calibrated real camera (exact K, T) and free orbit camera. Show camera name/pose, active resolution and render FPS. State clearly when a free orbit view uses a chosen virtual K rather than a real calibrated sensor's intrinsics.
- Rendering modes: RGB (default), alpha, depth (with metric units/colorbar), Gaussian link-ID visualization; optional SS/SH degree and quality/resolution controls with conservative defaults for RTX 4070 8 GB.
- Debug overlays toggles: URDF link frames, joint axes, hinge points, bounding boxes, collision geometry as **overlays only** (not replacing Gaussians), link name and q readout. Overlay coordinate systems must move with actual SAPIEN link poses.
- Screenshot/export button: save current RGB, depth, alpha, camera K/c2w, all q and `T_world_link` to `outputs/<run_id>/`; video recording may be a later optional feature.
- Show validation/error messages inside GUI: missing PLY, wrong frame, unbounded joint, failed gsplat call, no visible alpha, CUDA OOM, stale camera/task state.
- No artificial robot, no reconstructed table/environment Gaussian assets. A plain configurable RGB background is sufficient. If a built-in floor/grid debug helper is shown, label it debug-only and do not treat it as a real 3DGS environment.

### Render synchronization and concurrency (critical)

Use one authoritative simulation state with a thread-safe render snapshot, **not** the GUI directly writing q while a render thread reads link transforms.

1. UI event (`slider`, `play`, `step`) enqueues desired state action with a revision ID.
2. Simulation/kinematic owner applies it using ManiSkill on one thread, synchronizes GPU pose when necessary, publishes immutable snapshot.
3. GUI calls `viewer.rerender(...)` after snapshot updates (verify exact `nerfview` version API). Existing camera `on_update` naturally triggers renders.
4. Render callback reads a **consistent snapshot** and **current camera state**, updates world Gaussians once per snapshot, invokes gsplat once, returns RGB (and optional depth) in the API shape expected by that pinned nerfview version.
5. Throttle/coalesce rapid camera movement and slider events; stale render requests must not overwrite the latest state. No unbounded queue, no busy-loop on a clientless server.
6. Each viewer client has independent orbit camera; when multiple clients are connected, specify whether joint slider state is shared or per-client. Default shared articulation q, per-client cameras; communicate shared state in GUI.
7. At shutdown, stop threads cleanly, close ManiSkill scene and viewer server, flush requested exports.

### Camera and overlay correctness

Viser camera orientation uses OpenCV axes according to its current published conventions. `nerfview.CameraState.c2w` must be inspected for the pinned version; use that c2w in the `gsplat` rasterization view matrix. For free orbit choose consistent virtual pinhole intrinsics based on viewport size/FOV. Browser-camera `K` may not support an arbitrary calibrated `cx,cy` offset: in *calibrated preset mode* render with the stored exact K, and don't claim it is the same as the unconstrained orbit camera. If an exact offset cannot be preserved through `nerfview`, present the calibrated frame in a dedicated view panel generated from true K, leaving free orbit separate.

### Suggested CLI acceptance invocation

```bash
conda activate parttwin_sim
cd ~/PArtTwin1D
source scripts/activate_sim.sh
python -m parttwin1d.cli.gui \
  --config configs/examples/hinged_box.yaml \
  --host 127.0.0.1 --port 8080
# Open http://127.0.0.1:8080
```

CLI must start without needing `pip install parttwin_smoke` or entering `legacy/`.

---

## 7. CONFIGURATION CONTRACT: REAL ARTICULATED OBJECTS

Provide YAML schema (JSON supported if already present) with required fields and explicit examples:

```yaml
schema_version: 1
object_name: red_fridge
urdf: ../../assets/red_fridge/model.urdf
fixed_base: true
T_world_root:                       # 4x4 root pose; camera/world frame carefully documented
  - [1, 0, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 0]
  - [0, 0, 0, 1]
links:
  base_link:
    ply: ../../assets/red_fridge/base.ply
    T_link_ply: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  door_link:
    ply: ../../assets/red_fridge/door.ply
    T_link_ply: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
cameras:
  - name: cam_000
    width: 640
    height: 480
    K: [[600.0, 0.0, 320.0], [0.0, 590.0, 240.0], [0.0, 0.0, 1.0]]
    T_world_camera: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]]
render:
  background_rgb: [1.0, 1.0, 1.0]
  sh_degree: -1
viewer:
  port: 8080
  preview_width: 640
  preview_height: 480
```

**This is a schema example, not validated red-fridge calibration.** For actual real scenes, Codex must read measured K, camera extrinsics, real joint axis/limit and PLY coordinate frame from the user's exports; **do not substitute these example numbers or invent asset paths**. Explain when `T_world_root`, PLY alignment, and camera frame share different origin conventions. Provide a manifest inspector with detailed field-by-field diagnostics.

---

## 8. TEST MATRIX: REAL INTEGRATION, NOT MOCKED SUCCESS

### Unit tests, always runnable on CPU

- `test_se3.py`: inverse/compose point transforms, covariance rotation, quaternion order vs rotation-matrix reference.
- `test_cameras.py`: `T_world_camera` projection and known positive/negative optical axes; `fx != fy`, `cx != W/2`, `cy != H/2`; validate rejection of unsupported skew/distortion, if necessary.
- `test_ply.py`: SH field layout degree 0–3, log-scale/opacity conversion, zero quaternion rejection, deterministic output, missing fields failure.
- `test_asset_schema.py`: URDF link/joint binding, missing PLY, duplicate links, wrong units/frame shape, unknown config keys, path resolution independent of CWD.
- `test_imports.py`: `parttwin1d` only; `python -m pytest` from root collects without `parttwin_smoke`/ROS Jazzy plugin errors; detect stale imports from Trash.
- `test_splat_link_pose.py`: moving means/orientations and static-link invariance under known revolute/prismatic motions; return to original pose has no drift.

### Real PhysX integration, no mocks; run on operator's workstation

- Load a tracked two-link hinged-box URDF through **installed ManiSkill 3.0.1 / SAPIEN 3.0.3**; no robot agent.
- Assert fixed link world pose unchanged, moving link rotation matches joint input and URDF pivot within numeric tolerance (e.g. rotation error <1e-4 rad for CPU kinematic preview), q and joint limits correct.
- Verify clean reset/reinitialize, `get_state_dict`, `get_obs`, update q, multiple q changes, cleanup, physics stepping separately.
- Repeat in `physx_cuda` if environment supports it; verify GPU apply/update/fetch semantics, not only CPU simulation.

### Real gsplat integration, no mocks

- Launch a **real** `gsplat.rendering.rasterization()` kernel; fail if fallback rendering is used or CUDA kernel unavailable.
- Inspect nonempty alpha and finite RGB; physically plausible positive Z where foreground visible; images change as joint moves.
- Verify the rendered static base is unchanged in world position and link transforms, while door Gaussian positions change.
- Two differently colored links with geometric overlap test proper depth occlusion, independent of draw/PLY iteration order.
- Confirm RGB/images saved and reloadable; GUI preview and offline render with same q/K/pose agree within tolerance (allow reasonable transport JPEG differences only if JPEG is used).
- Distinguish **compile-time** from **steady-state frame latency**; run warm-up frames and log FPS, peak GPU memory, total Gaussians, and H×W.

### GUI manual + automated interaction test

- GUI opens in real browser at declared URL, renders visible **3D Gaussian RGB** without first exporting any file, and accepts mouse orbit, pan and zoom.
- Drag joint slider 0 → +1.0 rad → 0; visible moving link follows pivot, static link unchanged, no accumulated drift, q display synchronized.
- Camera orbit triggers **new** gsplat render with new K/c2w; not just rotating a pre-rendered image plane.
- Play/pause actually changes/respects physical q state; export writes actual K, c2w, q, link poses, RGB/depth/alpha.
- Toggle axis and frame overlays; verify overlays track the simulated joint accurately and do not replace splats.
- Test two browser clients to reveal shared-state races; test browser disconnect/reconnect and server Ctrl+C shutdown.
- All GUI errors must be surfaced, not swallowed by a background worker.

### Required test commands and result capture

```bash
cd ~/PArtTwin1D
source scripts/activate_sim.sh
python -m pytest -q             # Must collect EVERY production test cleanly
python -m parttwin1d.cli.doctor --provenance
PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/integration
python -m parttwin1d.cli.render --config configs/examples/hinged_box.yaml \
  --joint door_hinge --q-start 0 --q-end 1.3 --frames 20 --out outputs/hinged_box
python -m parttwin1d.cli.gui --config configs/examples/hinged_box.yaml --port 8080
```

Provide evidence with actual commands, stdout/stderr, exit codes, images, and paths. It is acceptable to report **UNTESTED** when Codex lacks a GPU/GUI runtime; it is unacceptable to present skipped or mocked tests as passes.

---

## 9. IMPLEMENTATION ORDER (FOLLOW IN THIS ORDER)

1. **Audit + cleanup:** document current tree and failures; protect work; create feature branch; move stale smoke artifacts/tests out of production imports.
2. **Submodules + dependency provenance:** add pinned ManiSkill, gsplat, nerfview source submodules; record SHAs/licenses; verify installed versions and API contracts. Do not reinstall Torch.
3. **Unified `src/` architecture:** migrate and rationalize existing modules; make packaging/test import robust; fix every root `pytest` collection error.
4. **Engine correctness:** real robotless ManiSkill URDF import, joint mapping, initial root pose, PhysX link state snapshot; independent CPU/GPU tests.
5. **Renderer correctness:** immutable canonical splats, exact per-link SE3, SH handling, one-pass occlusion, exact K, GPU outputs; actual rasterization tests.
6. **Interactive GUI:** integrate pinned nerfview/viser, live orbit and joint controls, physical/kinematic modes, overlays, exports, background worker safety; verify browser experience.
7. **Real asset path:** load one actual user's real-object export (if available locally), inspect alignment and demonstrate physically correct visual movement; never invent calibration data.
8. **Measure, document, commit:** test matrix, reproducible setup, profile, explicit known limitations, commit cohesive changes to the feature branch. Avoid pushing or opening PRs without authorization/workflow requirement.

Keep work focused on production code. No training of 3DGS, no new custom CUDA rasterizer, no full reconstructed robot/environment, and **no PPO training yet**. Preserve a future interface `ArticulatedGaussianRenderer.render_observation(snapshot, cameras) -> GPU tensors` suitable for a ManiSkill visual PPO environment. Do not claim robot/3DGS depth compositing or parallel PPO are implemented before they have real tests.

---

## 10. FINAL CODEX REPORT FORMAT (MANDATORY)

On completion write `docs/IMPLEMENTATION_REPORT.md` containing:

1. **Before:** root cause of stale `parttwin_smoke` test imports; actual prior render stall/failure; exact files/lines involved.
2. **After:** tree of created/moved files; changed API boundaries; submodule URLs + full pinned commit SHAs + licenses; package and CUDA source provenance.
3. **Tests:** table with per-test command, **PASS / FAIL / SKIPPED / NOT RUN**, exit status, observed evidence. Separate mocked unit tests, physical ManiSkill tests, actual gsplat CUDA tests and GUI manual tests.
4. **GUI:** launch command and URL; screenshot showing real Gaussian RGB in interactive view and controls; known browser/camera limitations; demonstrated hinge rotation.
5. **Measurements:** image resolution, Gaussian count, fps (after warm-up), peak GPU memory, per-link alignment errors, render latency and scene type.
6. **Incomplete:** remaining bugs and exact reproduction commands; no vague "works" or untested declarations.
7. **Operator commands:** at most one setup/bootstrap command and one GUI launch command, from `~/PArtTwin1D`, with no disposable smoke package.

**Stop only when the repository is clean, all production test collection succeeds, the real SAPIEN and gsplat integration has been exercised, and the 3D GUI is actually interactive. If workstation-specific execution is unavailable, implement everything possible and report the runtime acceptance items as outstanding, explicitly.**

## Verified external API references to inspect while coding

- ManiSkill 3.0.1 release: https://github.com/mani-skill/ManiSkill/releases/tag/v3.0.1
- ManiSkill `BaseEnv` and URDF loader source (pin `v3.0.1`, not `main`): https://github.com/mani-skill/ManiSkill
- gsplat 1.5.3 release: https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.3
- gsplat 1.5.3 rendering signature: https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py
- nerfview repository with `Viewer`, `Viewer.rerender`, camera-state callback: https://github.com/nerfstudio-project/nerfview
- Viser camera coordinate convention (+Z forward, +Y down): https://viser.studio/main/conventions/
- Viser GUI and camera APIs: https://viser.studio/main/api/core/server/
- gsplat's own interactive viewer reference: https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_viewer.py (reference only; match pinned gsplat API when implementing)

**Explicit user directive:** Work inside the pre-existing `~/PArtTwin1D` checkout and `parttwin_sim`. Build a structured, sustainable, self-contained-in-source repository with pinned cloned submodules and a real 3D GUI. No more `smoke` packages.
