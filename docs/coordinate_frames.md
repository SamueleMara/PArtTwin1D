# Coordinate frames and the mathematical contract

## Notation

`T_A_B` (in maths `T_{A←B}`) is a 4×4 proper rigid transform mapping homogeneous
column vectors expressed in frame **B** into frame **A**:

```
[x_A; 1] = T_A_B [x_B; 1]          T_A_C = T_A_B · T_B_C          T_B_A = T_A_B⁻¹
```

Every transform is validated (`geometry.se3.validate_se3`): 4×4, finite,
bottom row `[0 0 0 1]`, `RᵀR = I` (tol 1e-5, i.e. **no scale/shear**) and
`det R = +1` (no reflections). Nothing is ever recentred, rescaled or axis-swapped
implicitly.

| Frame | Meaning |
|---|---|
| `W` | simulator world (ManiSkill/SAPIEN), z-up, metres |
| `L_i` | URDF link frame of link *i* (for a joint: child link frame = joint frame) |
| `P_i` | canonical frame of link *i*'s Gaussian PLY |
| `C` | OpenCV optical camera: +X right, +Y down, +Z forward |
| `Wc` | capture/reconstruction world (e.g. COLMAP), only for alignment |

## Per-link Gaussian transport (renderer)

```
T_W_Pi(q) = T_W_Li^PhysX(q) · T_Li_Pi
```

* `T_W_Li^PhysX(q)` — **read from the live SAPIEN articulation** (`Link.pose`); never
  computed by a home-made FK.
* `T_Li_Pi` — `links.<name>.T_link_ply` in the config, **required** (state identity
  explicitly).

With `R, t` the blocks of `T_W_Pi`:

| Quantity | Transform |
|---|---|
| mean | `μ_W = R μ_P + t` |
| covariance | `Σ_W = R Σ_P Rᵀ` (implemented through the quaternion, below) |
| orientation | `q_W = q(R) ⊗ q_P` — scalar-first `(w,x,y,z)`, Hamilton product, so `R(q_W) = R · R(q_P)` — gsplat's convention |
| scale / opacity | unchanged |
| view-dependent SH | evaluated in the **canonical frame P**: `d_P = R⁻¹(μ_W − c_W) = μ_P − R⁻¹(c_W − t)`; colour `= max(SH(d_P) + 0.5, 0)` passed to gsplat as RGB (`sh_degree=None`) |

Why the SH rule: gsplat's built-in SH path evaluates `d_W = μ_W − c_W` in the
**world** basis. For a rotated link that rotates the appearance relative to the
geometry. Tests (`tests/integration/test_gsplat_render.py`):

* `test_sh_rigid_invariance[0,1,3]` — moving the link by `G` and the camera by `G`
  gives the same image (max |ΔRGB| < 2e-3) for SH degree 0, 1, 3;
* `test_world_frame_sh_would_be_wrong` — the built-in world-frame path fails the
  same check (negative control);
* `test_canonical_sh_equals_gsplat_reference_at_identity` — at `T_W_P = I` our
  colours reproduce gsplat's own SH evaluation (same sign: `d = μ − c`).

Canonical buffers are uploaded once and never written; world buffers are
recomputed from them for every snapshot revision → returning to q0 is
bit-identical (`test_no_drift_and_canonical_immutable`).

## Cameras

* Stored per calibrated camera: `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, `width`,
  `height`, `T_world_camera` = **c2w in OpenCV optical axes**.
* gsplat receives `viewmats = T_world_camera⁻¹` (w2c) and `Ks [C,3,3]`.
* **Skew is rejected**: gsplat 1.5.3's projection kernels only read
  `Ks[0], Ks[4], Ks[2], Ks[5]` (fx, fy, cx, cy), a non-zero `K[0,1]` would be
  silently ignored. Lens distortion must be removed upstream (rectified K).
* Resizing a calibrated camera scales K exactly: `fx·sx, fy·sy, cx·sx, cy·sy`.
* **Viser/nerfview** cameras are already OpenCV (`+Z` forward, `+Y` down; see
  `CameraHandle.wxyz` docstring in viser 1.1.1). No OpenGL flip is applied —
  verified by the RGB axis triad in the fixture (red +X, green +Y, blue +Z up in
  the GUI screenshots) and by `test_optical_axes_positive_and_negative`.
* GUI orbit cameras use a **virtual** K (`fx = fy = H/2/tan(fov_y/2)`, centred
  principal point — identical to `nerfview.CameraState.get_K`). Calibrated
  intrinsics are only used in calibrated mode (offline render, exports, the
  *Calibrated view* panel), never claimed for the orbit view.

## Root pose

`T_world_root` (required) is applied to the ManiSkill articulation builder's
`initial_pose` **before** PhysX initialisation (`sim/env.py`), and verified after
reset and after reconfigure (`root_pose_error < 1e-5`).

## Alignment from a capture world

If part *i*'s Gaussians are in capture-world coordinates at reference q0 and the
registration `T_W_Wc` (rigid, metres) is known:

```
T_Li_Pi = T_W_Li(q0)⁻¹ · T_W_Wc · T_Wc_Pi
```

(`assets.frame_alignment.link_ply_from_capture`). It must be derived
algebraically from calibrated quantities, then checked with
`collision_containment` (fraction of Gaussian means inside the link's URDF
collision primitives; `parttwin inspect` reports it per link). If the
reconstruction is not metric, rescale the PLY explicitly first — scale is never
hidden in a 4×4.

## Depth

`depth` = gsplat `RGB+ED` expected optical-Z depth in metres
(`Σ wᵢ zᵢ / Σ wᵢ`), `NaN` where `alpha < alpha_threshold` (default 0.01), with a
boolean `valid` mask. It is not a hard surface or collision depth.
