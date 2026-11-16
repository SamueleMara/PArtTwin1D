# PArtTwin1D ManiSkill + gsplat v0.3.0 — reproducible user commands

This release fixes the **robot-free ManiSkill `BaseEnv` crash** seen on
ManiSkill 3.0.1 / SAPIEN 3.0.3 and installs in a **new directory** to avoid the
old editable install that was resolving under `~/.local/share/Trash/files/`.

## Extract into the existing repository

Open a **new terminal** and run (use the real location of the downloaded ZIP):

```bash
conda activate parttwin_sim
cd "$HOME/PArtTwin1D"
unzip -q PArtTwin1D_ManiSkill_3DGS_v3_Robotless_Fix.zip
cd "$HOME/PArtTwin1D/parttwin_maniskill_gs_v3"
pwd -P
source ./setup_env.sh

python -m pip uninstall -y parttwin-maniskill-gs-smoke
python -m pip install -e . --no-deps
python - <<'PY'
from pathlib import Path
import parttwin_smoke
p = Path(parttwin_smoke.__file__).resolve()
print('ACTIVE PARTTWIN CODE:', p)
assert 'parttwin_maniskill_gs_v3' in str(p), p
assert '.local/share/Trash' not in str(p), p
PY

python -m pytest -q tests
```

If the ZIP is in `~/Downloads`, use `unzip -q ~/Downloads/PArtTwin1D_ManiSkill_3DGS_v3_Robotless_Fix.zip` while in `$HOME/PArtTwin1D`.

## CUDA rasterizer: actual test, not just import

```bash
# The first real call to gsplat may JIT-compile its CUDA extension.
# Do not interrupt just because it shows `TypeError: _jit_compile()` in a
# preliminary fallback traceback: gsplat 1.5.3 retries with another signature.
# If this command fails, save the full log and show the final traceback.
mkdir -p outputs
set -o pipefail
python -u -m parttwin_smoke.doctor --output outputs/gsplat_preflight.png \
  2>&1 | tee outputs/gsplat_doctor.log
```

The command should print **`Actual CUDA rasterization: PASSED`** and create
`outputs/gsplat_preflight.png`. An `ImportError: cannot import name 'csrc'`
is part of gsplat's JIT loading path; neither that message nor the internal
`TypeError: _jit_compile()` establishes an actual CUDA failure. A user-issued
`KeyboardInterrupt` also does not prove compilation failed.

If a previously compiled gsplat extension triggers a full rebuild, verify that
`CUDA_HOME` and `nvcc` are still on 12.8 and that you are using the same
`TORCH_EXTENSIONS_DIR` (if any) as the previous successful build. The setup
script intentionally does not change this variable.

## Actual ManiSkill articulated-URDF + 3DGS animation

```bash
python -m parttwin_smoke.run \
  --manifest examples/synthetic_fixture/asset.json \
  --q-min 0 --q-max 1.3 --frames 20 \
  --output outputs/synthetic_door
```

Expected: `outputs/synthetic_door/sweep.gif`, PNG frames, depth+alpha NPZ files,
`report.json`. This uses ManiSkill/PhysX to compute every link pose, and gsplat
to render **only** the Gaussian parts; there is no robot or reconstructed room.
It is a prescribed joint sweep, **not yet physics force-driven motion or PPO**.

For a real moving Gaussian part in the matching link-local frame:

```bash
python -m parttwin_smoke.run \
  --urdf /absolute/path/to/model.urdf \
  --ply /absolute/path/to/moving_part.ply \
  --link door_link --joint-name door_hinge \
  --q-min 0 --q-max 1.3 --frames 30 \
  --output outputs/real_door
```

To include stationary and moving parts in one render, edit a JSON manifest
matching each original PLY to the correct named URDF link. Provide `T_link_ply`
for PLYs that are not already in the link-local frame.

## Explanations of the errors from 16 November 2026

1. **`cd: ... No such file or directory`**: the old folder is no longer at the path;
   Python imported `parttwin_smoke` from Trash. New v3 directory plus a fresh
   editable installation fixes that without deleting anything from Trash.
2. **`AttributeError: 'NoneType' object has no attribute 'get_controller_state'`**:
   the previous v2 subclass used `robot_uids="none"` but inherited ManiSkill's
   agent-dependent `get_state_dict`. v3 overrides it and returns
   `self.scene.get_sim_state()`; also sets an explicit root pose.
3. **`KeyboardInterrupt` in `ninja`**: the user interrupted gsplat's CUDA
   compilation. This is not evidence of a build failure. The `doctor` command
   only passes when real rasterization succeeds.
4. **ROS `launch_pytest` missing `lark`**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
   stops ROS plugins from contaminating pytest in the Python 3.11 Conda env.

## Validation status

- **CPU/mock tests**: passed in authoring container (11 tests, including a
  robot-free ManiSkill constructor compatibility regression using a fake
  BaseEnv). These do not count as real ManiSkill integration tests.
- **Real ManiSkill 3.0.1 + PhysX + CUDA gsplat**: must be run in the user's
  `parttwin_sim` environment; there is no real ManiSkill/gsplat GPU backend in
  the authoring container. Avoid claiming full system success from pytest alone.

References: [ManiSkill environment implementation](https://github.com/mani-skill/ManiSkill),
[ManiSkill URDF loader](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/building/urdf_loader.py),
[gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3).
