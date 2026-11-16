# URDF + per-link Gaussian PLYs

## Inputs

* **URDF** (metres, radians): one root link, fixed and movable joints with
  `<limit lower upper>` for revolute/prismatic, collision geometry and
  `<inertial>` where known. Loaded by **ManiSkill's URDF loader** into one PhysX
  articulation; URDF `<visual>` records are dropped (appearance = Gaussians).
* **One Graphdeco 3DGS PLY per rendered link** (`x y z opacity f_dc_0..2
  f_rest_* scale_0..2 rot_0..3`, `f_rest` channel-major). Degree 0–3 supported
  per link (mixed degrees are zero-padded). All Gaussians are kept.
  Plain XYZ/RGB point clouds are rejected.
* **`T_link_ply`** per link (required) — see `coordinate_frames.md`.

## Config (`schema_version: 1`)

See `configs/examples/hinged_box.yaml` (synthetic fixture) and
`configs/real_object.template.yaml` (placeholders only). Required:
`schema_version, object_name, urdf, T_world_root, links.<link>.{ply,T_link_ply}`.
Optional: `fixed_base` (default true), `initial_q`, `cameras[]`, `render`,
`viewer`, `sim`. Unknown keys are errors; relative paths resolve against the
config file.

## Joint semantics

* Active DoF are discovered **by name** from `Articulation.get_active_joints()`
  and `get_qlimits()` (PhysX order), cross-checked against the URDF; fixed and
  mimic joints are not DoF.
* `initial_q` defaults to 0 clipped into the limits.
* **Kinematic preview** sets q (and zero qvel) — no time passes.
* **Dynamics preview** steps PhysX (`env.step(None)`) under an explicit input:
  PD joint drive target (`sim.drive_stiffness/damping/force_limit`, set before
  GPU init) or a generalized joint force; with a force input the drive target
  tracks q (pure damping). Gravity, URDF damping/friction and inertia apply.

## Checking a real export

```bash
python -m parttwin1d.cli.inspect --config assets/<obj>/object.yaml
```

Field-by-field report: URDF joints (type, axis, limits, damping/friction),
links without collision/inertial data, PLY Gaussian counts / SH degree / AABB,
implausible extents (unit errors), **percentage of Gaussian means inside each
link's collision primitives** (frame/scale check), `initial_q` limits, camera
intrinsics and whether the object projects into each camera.
