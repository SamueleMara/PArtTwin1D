# PArtTwin1D — ManiSkill/SAPIEN × articulated 3DGS

This is a repository-level integration, not a separately installed package. It
imports an articulated URDF into **real SAPIEN/PhysX** and renders the existing
canonical per-link 3DGS PLYs with **gsplat**. No reconstructed background, no
robot Gaussian model, no PPO yet. No proprietary renderer, no custom rasterizer.

## Architecture

```text
URDF (joints + collision meshes) → ManiSkill Articulation
                                      │
                 live PhysX Link.pose per link (T_world_link)
                                      │
              per-link T_link_ply + canonical Gaussian PLY
                                      │
                      T_world_ply = T_world_link T_link_ply
                                      │
                     gsplat GPU RGB + alpha + expected depth
                                      │
                        render_tensor()   /   render_numpy()
                        GPU observations      debug file export
```

**Keep appearance, collisions and learned dynamics separate.** The URDF
collision/inertial shapes are NOT generated from the PLY by this code. Missing
collision meshes must be supplied/constructed separately. `render_backend="none"`
disables only the native Vulkan image renderer, not PhysX.

## Use your existing installation

Already installed in the user's `parttwin_sim` environment: ManiSkill 3.0.1,
SAPIEN 3.0.3, torch 2.11+cu128, gsplat 1.5.3, numpy/scipy/plyfile/Pillow/pytest.
No additional package installation is needed.

```bash
conda activate parttwin_sim
cd ~/PArtTwin1D
source scripts/activate_sim.sh

# Check imports/compiler (does not trigger gsplat JIT)
python scripts/check_sim.py

# CPU/unit regressions (avoids ROS 2 pytest plugin conflict)
python -m pytest -q tests

# Optional generated 2-link fixture, only to validate interfaces
python scripts/create_demo_assets.py
python scripts/inspect_asset.py --config configs/hinged_box.json

# Real PhysX articulation and actual gsplat RGB images
python scripts/render_articulated.py \
  --config configs/hinged_box.json --joint door_hinge \
  --q-start 0 --q-end 1.3 --frames 20 \
  --out outputs/hinged_box
```

**Important:** The first real call to gsplat may trigger several minutes of CUDA
extension compilation. It is not a hang. `source scripts/activate_sim.sh` deliberately does not change `TORCH_EXTENSIONS_DIR`,
so Torch can reuse a previously compiled gsplat extension; do not interrupt unless compilation fails.
An internal `_jit_compile` TypeError in gsplat 1.5.3/torch 2.11 is caught by its
fallback and is not itself the terminal build failure.

View `outputs/hinged_box/front_oblique.gif` and check
`outputs/hinged_box/report.json`. The per-frame PNG and depth/alpha NPZ files
are under the camera subfolder. A blank image or no visual motion is a failure.

## Load YOUR reconstructed articulated object

Set up local files:

```text
PArtTwin1D/
  assets/my_object/
    model.urdf            # physically valid articulated URDF + collisions
    base.ply              # actual trained Gaussian PLY of fixed link
    moving.ply            # actual trained Gaussian PLY of moving link
  configs/my_object.json  # derived from configs/real_object.template.json
```

Set `links` keys **exactly** to `articulation.links_map` link names. Set the true
`T_link_ply` for each PLY. `T_A_B` maps coordinates expressed in B to A. Thus:

\[
T_{world,ply}(q) = T_{world,link}(q)\,T_{link,ply}.
\]

If Gaussian xyz are already expressed in the matching URDF link frame,
`T_link_ply=I`. If PLYs are in a common reconstruction-world reference frame
at an initial pose `q0` and the reconstruction has been aligned to simulator
world, then `T_link_ply = inverse(T_world_link(q0)) * T_world_ply(q0)`.
Do **not** set this to identity merely because the object looks approximately
aligned. No implicit centering/scaling is performed; Gaussian coordinates and
URDF lengths must both be in meters.

Set camera `K` and `T_world_camera` in OpenCV optical convention:
+X to the right, +Y down, +Z forward. `T_world_camera` is camera → world,
not world → camera. The renderer uses the exact fx, fy, cx, cy.

```bash
python scripts/inspect_asset.py --config configs/my_object.json
python scripts/render_articulated.py \
  --config configs/my_object.json \
  --joint YOUR_JOINT_NAME --q-start 0 --q-end 1.0 --frames 32 \
  --out outputs/my_object
```

Revolute q values are radians; prismatic values are meters. Values must be
inside the URDF joint limits. For multiple joints, the driver changes the
selected joint and leaves others at their initialized positions.

## Real engine integration (not just image export)

Use the reusable renderer inside any ManiSkill task with its *existing*
articulated object:

```python
from parttwin1d.sim.assets import load_config
from parttwin1d.sim.renderer import ArticulatedGaussianRenderer

cfg = load_config("configs/my_object.json")
gs = ArticulatedGaussianRenderer(cfg, device="cuda")
# `articulation` is a ManiSkill Articulation in any existing scene
rgb_obs = gs.render_tensor(articulation, cfg.cameras[0])["rgb"]  # CUDA H,W,3
```

It consumes **live** link poses; the renderer does not set joints or drive
physics. `render_numpy` is for saving/debugging and causes GPU→CPU transfers.
This is the intended interface to a future ManiSkill RGB observation adapter
for PPO. When adding a robot, depth-correct robot/3DGS compositing must be
implemented and tested; simply overlaying RGBA images is not sufficient.

## Runtime boundaries and regression requirements

- PhysX alone owns state, link poses, contacts, joint limits.
- One PLY per bound link; explicit rigid `T_link_ply` transform.
- Raw Graphdeco/3DGS PLY required (`f_dc`, `f_rest`, opacity logits, log-scales,
  scalar-first quaternion). XYZ/RGB point clouds are **not** Gaussian assets.
- SH is evaluated in the original PLY coordinates for moving links, avoiding
  rotating view-dependent color incorrectly.
- Exact calibrated camera matrices and support for multiple cameras.
- Output RGB stays on CUDA until explicit file export.
- No hard-coded robot/table, no reimplementation of Gaussian rasterization.
- Unit tests are NOT an integration pass. A full GPU test requires actual PhysX
  import, actual gsplat kernel launch, nonempty alpha, and different RGB frames.

Recommended next engineering steps after a successful real asset test: add a
robot-equipped ManiSkill task with collision geometry and object-only Gaussian
camera observations; then add robot/3DGS depth compositing and vectorized
rendering, and only then run RGB-PPO.
