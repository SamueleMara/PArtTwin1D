# PArtTwin1D — minimal articulated ManiSkill + 3DGS smoke test

**Version: v0.2.0 — revised for ManiSkill 3.0.1, SAPIEN 3.0.3, and gsplat 1.5.3 (16 November 2026).**

**Scope:** An actual ManiSkill 3 / SAPIEN articulation loaded from your existing URDF, with **one or more raw graphdeco-format Gaussian PLY files mapped to actual URDF links**. The only visual renderer is the existing `gsplat` CUDA rasterizer. It produces image frames, depth/alpha, a GIF, and link-pose diagnostics. **No robot, no background reconstruction, no robot Gaussians, no PPO, no training or parameter estimation.**

The state owner is ManiSkill's PhysX articulation. The smoke test **sets qpos directly**, so this demonstrates correct imported kinematics and rendering, not identified dynamics.

## Installation (Ubuntu + NVIDIA CUDA)

Inside a new environment with a compatible CUDA-enabled PyTorch installation:

```bash
# Existing parttwin_sim environment already has ManiSkill, gsplat and CUDA.
conda activate parttwin_sim
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
python -m pip install -e . --no-deps
python -m parttwin_smoke.doctor --output outputs/gsplat_preflight.png
```

The package pins the Python-facing interfaces to the installed environment (ManiSkill 3.0.1 and gsplat 1.5.3). Run `python -m pip check` first. A first actual gsplat kernel render must pass `python -m parttwin_smoke.doctor` before testing your URDF. See `INSTALL_UBUNTU.md` for the full environment/setup sequence.

**Caution:** In the authoring environment only CPU regression tests could be run. An actual ManiSkill + CUDA smoke test must run on your RTX 4070 machine.

## First smoke: synthetic fixture (diagnostic only)

```bash
# The ZIP already includes the fixture; regenerate only if needed:
python examples/create_synthetic_fixture.py --output /tmp/parttwin_synthetic
python -m parttwin_smoke.run \
  --manifest /tmp/parttwin_synthetic/asset.json \
  --q-min 0 --q-max 1.3 --frames 16 \
  --output /tmp/parttwin_synthetic/output
```

Inspect `/tmp/parttwin_synthetic/output/sweep.gif`, the numbered RGB PNGs, per-frame depth/alpha NPZ, and `report.json`. The static frame remains stationary while the red door swings about its z-axis. This synthetic asset is only a **smoke test** and is not a reconstruction result.

## Real articulated object: two or more PLYs

Example `assets/fridge/asset.json` (paths relative to the JSON file):

```json
{
  "urdf": "model.urdf",
  "links": {
    "base_link": {"ply": "gaussians/base_link.ply"},
    "door_link": {"ply": "gaussians/door_link.ply"}
  }
}
```

```bash
python -m parttwin_smoke.run \
  --manifest /absolute/path/to/asset.json \
  --q-min 0 --q-max 1.4 --frames 24 \
  --width 640 --height 480 \
  --output outputs/my_fridge_smoke
```

**Coordinate contract:** each PLY above must already be expressed in its declared **URDF link-local frame**. If your Gaussian export is in a different canonical frame, provide each link's `T_link_ply` (a row-major 4x4 transform mapping PLY xyz into the URDF link frame):

```json
"door_link": {
  "ply": "gaussians/door_original.ply",
  "T_link_ply": [[1,0,0,0.01],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

This transform must preserve scale, or the PLY/URDF must be converted to matching metres before use. The code validates that `T_link_ply` is rigid (rotation + translation). **No arbitrary centering and no guessed hinge transforms.**

### If you currently have only ONE PLY

To prove the moving-link render first, bind it to one existing URDF link:

```bash
python -m parttwin_smoke.run \
  --urdf /absolute/path/to/model.urdf \
  --ply /absolute/path/to/moving_part.ply \
  --link door_link --q-min 0 --q-max 1.2 \
  --output outputs/door_only
```

This is correct **only when the PLY coordinates are already local to `door_link`**. If your PLY is in the *simulator-aligned world frame at q_min* instead, add `--ply-in-reference-world` and verify the resulting alignment. If it is in an unrelated reconstruction/COLMAP frame, first calculate the exact `T_link_ply` and use the manifest.

If your single PLY contains *both* static and moving Gaussians, their part memberships must be recovered/exported first; one rigid transform cannot articulate the entire mixture correctly.

## Calibrated camera (optional)

Provide exact intrinsics and an OpenCV optical camera pose instead of assuming centered/square camera calibration:

```json
{
  "width": 640, "height": 480,
  "K": [[585.0,0,310.0],[0,579.0,236.0],[0,0,1]],
  "T_world_camera": [[1,0,0,0.2],[0,1,0,-0.5],[0,0,1,0.6],[0,0,0,1]]
}
```

`T_world_camera` maps **camera to simulator world**, with OpenCV optical axes (+Z forward, +X right, +Y down). The shown matrix is only an illustrative schema and may not aim at your object. Without a camera JSON, the script derives a framing camera from the transformed Gaussian bounding box. You can override with `--eye x y z --target x y z --fx ... --fy ... --cx ... --cy ...`.

## Validation / limitations

1. `pytest -q tests` exercises rigid frames, quaternion order, a z-axis hinge, SH coefficient ordering, camera forward, and asset contracts without any graphics dependency.
2. `create_synthetic_fixture.py` + smoke command verifies the complete CUDA + ManiSkill integration **on your machine**.
3. Compare a real PLY at q_ref to your original RigEuler RGB image using known camera calibration. Then check two different joint states and ensure the base is stationary, the door moves rigidly, and the hinge line remains fixed.
4. The renderer uses original PLY quaternions (WXYZ), exp(log-scales), sigmoid(opacity), SH in **the PLY's rotating local frame**. The gsplat 1.5.3 SH signature is `spherical_harmonics(degree, dirs[N,3], coeffs[N,K,3])`; the previous release passed the wrong argument shape, now fixed and regression-tested. SH degree 0 is a useful debug mode (`--sh-degree 0`); default `--sh-degree -1` uses available degrees.
5. Gaussian splats **do not produce collisions**. The URDF must contain safe collision/inertial metadata for physical tasks. Native ManiSkill meshes are not rendered in the smoke test; the background is white or black. No robot or world Gaussian is required.
6. `--q-min/max` use radians for revolute joints and **metres** for prismatic joints. The sweep directly sets joint qpos and reports exact SAPIEN link transformations; it is not an actuator / force-response benchmark.
7. Full GPU rendering has **not** been executed by the authoring environment (ManiSkill, gsplat and the user PLY/URDF are not installed here). The code is a testable scaffold, not a claim of a verified production integration.

## Reused upstream components

- [ManiSkill](https://github.com/mani-skill/ManiSkill): `BaseEnv`, URDF loader, PhysX articulation and link pose interfaces.
- [gsplat](https://github.com/nerfstudio-project/gsplat): CUDA `rasterization()` and SH evaluation. No custom Gaussian rasterizer.
- [ManiSkill-GS](https://github.com/chenkang455/ManiSkill-GS): useful reference for a future RL wrapper; its current actor logic assumes a rendered robot/table and its environment uses an older pinned ManiSkill beta, so it is intentionally not required for this minimal test.

### Next step after success

Replace the qpos sweep with physics stepping, then inject the resulting external GS RGB into ManiSkill observation data. Only after that should the PPO task, robot, environment observations and contact dynamics be added.

## Version 0.2.0 corrections

- Fixed `spherical_harmonics()` invocation to pass per-Gaussian direction vectors rather than a camera matrix. Added a regression test for SH direction changes as the hinge rotates.
- Added `python -m parttwin_smoke.doctor`, which performs a **real** simple CUDA rasterization and writes a diagnostic PNG.
- Added `--joint-name` to select the active URDF joint explicitly.
- Added explicit GPU articulation apply/fetch for optional `--sim-backend physx_cuda` and validation of declared link names before rendering.
- Depth pixels with alpha below 0.01 are exported as NaN (no visible splat).
- Kept CPU physics as the default: the geometry-only test does not require ManiSkill Vulkan or a robot asset. The GS renderer still requires CUDA.

### Shortest commands with existing packages

```bash
conda activate parttwin_sim
cd /path/to/parttwin_maniskill_gs_v2
python -m pip install -e . --no-deps
python -m pytest -q tests
python -m parttwin_smoke.doctor
python -m parttwin_smoke.run --manifest examples/synthetic_fixture/asset.json --frames 20 --output outputs/synthetic_door
```

**References:** gsplat [`v1.5.3` rendering and SH interfaces](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3), [ManiSkill docs](https://maniskill.readthedocs.io/en/latest/).

## Version 0.3.0 — robot-free ManiSkill fix

This release supersedes the v0.2.0 notes above. In ManiSkill 3.0.1,
`BaseEnv.get_state_dict()` calls `self.agent.get_controller_state()` even with
`robot_uids="none"`. The robot-free subclass now overrides this method to
return `self.scene.get_sim_state()` and overrides `_get_obs_agent` with an empty
dictionary. The URDF is built using ManiSkill's articulation builder with an
explicit initial pose, eliminating the default-pose warning. The optional GPU
path now calls `gpu_update_articulation_kinematics()` after applying qpos.

Run the commands in **INSTALL_UBUNTU.md**. The package must resolve from
`parttwin_maniskill_gs_v3`, never a removed v2 folder or a Trash path.
`setup_env.sh` configures CUDA 12.8 and suppresses ROS pytest plugin loading.
The gsplat `doctor` deliberately performs real CUDA rasterization and is not
covered by mock unit tests. The authoring environment passed 11 CPU tests;
end-to-end GPU rendering has not been executed here.
