# Architecture

```
config (YAML, schema v1) ─► URDF + per-link Gaussian PLYs + calibrated cameras
        │
        ▼
sim/env.py  ArticulatedTwinEnv(BaseEnv, robot_uids="none", render_backend="none")
        │      ManiSkill URDF loader → one PhysX articulation, explicit T_world_root
        ▼
sim/articulation.py  ArticulationDriver          sim/state.py
        kinematic: set_qpos (+GPU apply/update/fetch)   StateSnapshot (immutable, read-only arrays)
        dynamics:  env.step(None) with PD drive or qf   link_poses_tensor [B,L,4,4] (no host copy)
        │
        ▼ snapshot (q, dq, T_world_link per bound link, revision, sim time)
rendering/gaussians.py      canonical per-link buffers on GPU (uploaded once, never written)
rendering/link_adapter.py   T_W_P = T_W_L·T_L_P → world means/quats; canonical-frame SH → RGB
rendering/gsplat_backend.py ONE gsplat.rasterization call per camera batch (RGB [+link one-hot] + ED)
        │
        ├─► RenderResult (GPU tensors)  ─► rendering/observations.py  CPU export PNG/NPY/NPZ/JSON
        ├─► render_observation(snapshot, cameras) → batched GPU tensors (future PPO interface)
        ▼
viewer/  (live GUI)
  coordination.py   SimulationCoordinator — single sim-owner thread, coalesced inputs, snapshot publish
  render_callback.py nerfview render_fn: browser c2w + virtual K → gsplat → (uint8 RGB, depth)
  server.py         GuiApp: viser server + pinned nerfview Viewer + overlays + workers + exports
  controls.py       joints / playback / physics / camera / render / overlays / export / status
  overlays.py       viser scene graph parented to per-link frames (move with SAPIEN poses)
cli/  doctor · inspect · render · gui      (`parttwin <cmd>`, `python -m parttwin1d.cli.<cmd>`)
evaluation/  consistency checks, profiling (warm-up vs steady state)
```

## Single implementations

* SE(3)/quaternions: `geometry/se3.py` only. Cameras: `geometry/cameras.py`
  (+ `rendering/cameras.py` for the `RenderCamera` object).
* PLY decoding: `assets/gaussian_ply.py` only.
* SAPIEN state access: `sim/state.py` (`capture_snapshot`, `link_poses_tensor`).
* Rasterization: gsplat 1.5.3 CUDA kernels only — no custom rasterizer, no
  fallback renderer, no meshes or point clouds substituted for appearance.

## Threading model (GUI)

| Thread | Owns / does |
|---|---|
| `parttwin-sim-owner` | the ManiSkill env; applies commands; publishes snapshots; calls subscribers |
| nerfview `Renderer` (one per browser) | `render_fn` under nerfview's global `VIEWER_LOCK` |
| `parttwin-calibrated` | exact-K render of the selected calibrated camera (coalesced by revision) |
| `parttwin-status` | status text every 0.5 s (Event wait, no busy loop) |
| viser event loop / pool | GUI callbacks → `coordinator.submit_*` (never touch the env) |

Ordering: UI event → request id → owner applies (latest joint value wins; bounded
command queue of 128) → snapshot (revision, request_id) → overlays update, slider
sync, `viewer.safe_rerender()` → each client's nerfview thread renders the
**current** snapshot with **its own** camera (stale tasks are replaced, not queued).

Shared vs per-client state: articulation q is **shared** by all browsers; each
browser has an **independent** orbit camera. Stated in the GUI Status panel.

## Performance boundary

The UI/export path crosses GPU→CPU once per snapshot for L 4×4 link poses and
once per displayed frame (JPEG transport to the browser). The render hot path
(`render_batch`) keeps everything on the GPU. `link_poses_tensor` reads batched
link poses on the simulation device without host copies — the intended input
for a future vectorised visual-PPO observation; multi-env batching and PPO
throughput are **not** implemented or benchmarked.
