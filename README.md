# PArtTwin1D

Beyond Appearance: Physically Predictive Articulated Digital Twins from One Demonstration.

This repository loads a user-supplied **articulated URDF** into **ManiSkill 3 /
SAPIEN PhysX** (no robot), attaches **per-link trained 3D Gaussian PLYs**, renders
them with **gsplat**'s CUDA rasterizer from the **live PhysX link poses**, and
serves a **live interactive 3D browser GUI** (viser + nerfview) with joint
sliders, kinematic playback, physics stepping, overlays and exports.

## Quick start (existing `parttwin_sim` env)

```bash
conda activate parttwin_sim
cd ~/PArtTwin1D
bash scripts/bootstrap_submodules.sh            # once: submodules + editable installs + gsplat JIT
source scripts/activate_sim.sh                  # every new shell
python -m parttwin1d.cli.gui --config configs/examples/hinged_box.yaml --port 8080
# open http://127.0.0.1:8080

# or: Gaussians inside ManiSkill's own SAPIEN viewer (desktop window, zero-lag compositing)
python -m parttwin1d.cli.maniskill_gui --config configs/examples/hinged_box.yaml
```

Other commands (`parttwin <cmd>` is the same as `python -m parttwin1d.cli.<cmd>`):

```bash
python -m parttwin1d.cli.doctor --provenance --check-gsplat     # versions, import paths, SHAs, CUDA
python -m parttwin1d.cli.inspect --config configs/examples/hinged_box.yaml
python -m parttwin1d.cli.render --config configs/examples/hinged_box.yaml \
    --joint door_hinge --q-start 0 --q-end 1.3 --frames 20 --out outputs/hinged_box
python -m pytest -q                                              # unit tests (CPU)
PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/integration   # real PhysX + gsplat CUDA
PARTTWIN_RUN_GUI_TESTS=1 python -m pytest -v tests/gui           # real browser (system Chrome)
```

`make setup | doctor | test | test-gpu | gui | maniskill-gui | render | inspect | evidence` wrap the same.

## Layout

```
src/parttwin1d/   config/ assets/ geometry/ sim/ rendering/ viewer/ evaluation/ cli/
third_party/      ManiSkill @ v3.0.1, gsplat @ v1.5.3, nerfview @ 4538024 (git submodules)
configs/          examples/hinged_box.yaml, real_object.template.yaml, gui.defaults.yaml
examples/         hinged_box/ (tracked synthetic fixture), demo_articulated.py
tests/            unit/ (CPU), integration/ (real PhysX + CUDA), gui/ (real browser)
scripts/          activate_sim.sh, bootstrap_submodules.sh, check_environment.sh, test*.sh, run_gui.sh
docs/             architecture, installation, coordinate_frames, urdf_and_gaussians, gui, testing,
                  third_party, IMPLEMENTATION_REPORT
legacy/experiments/  archived smoke-era code (not importable, not collected)
```

## Using your own object

Copy `configs/real_object.template.yaml` next to your export, fill in the
**measured** URDF path, PLY paths, `T_world_root`, every `T_link_ply` and camera
calibration, then run `parttwin inspect` before the GUI. See
`docs/urdf_and_gaussians.md` and `docs/coordinate_frames.md`.

Scope today: one articulated object, no robot, no background Gaussians, no PPO.
