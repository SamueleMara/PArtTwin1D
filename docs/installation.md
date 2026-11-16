# Installation

Target: the **existing** `parttwin_sim` Conda env (Python 3.11.18, torch
2.11.0+cu128, SAPIEN 3.0.3, CUDA 12.8 toolkit from conda). Nothing here
recreates the env or reinstalls torch/CUDA.

## Fresh clone (other machine)

```bash
git clone --recurse-submodules git@github.com:SamueleMara/PArtTwin1D.git
cd PArtTwin1D
conda activate parttwin_sim          # or an env with the same torch/SAPIEN/CUDA versions
bash scripts/bootstrap_submodules.sh
```

## Existing checkout (`~/PArtTwin1D`)

```bash
conda activate parttwin_sim
cd ~/PArtTwin1D
bash scripts/bootstrap_submodules.sh   # = git submodule update --init --recursive
                                       #   + pip install --no-deps -e third_party/{ManiSkill,nerfview}
                                       #   + BUILD_NO_CUDA=1 pip install --no-deps -e third_party/gsplat
                                       #   + pip install --no-deps -e .
                                       #   + viser/splines pins, pip check, doctor
```

Every new shell: `conda activate parttwin_sim && source scripts/activate_sim.sh`.

`scripts/activate_sim.sh`:

* `unset PYTHONPATH` — ROS Jazzy / MoveIt Python 3.12 paths otherwise leak into
  this Python 3.11 env; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
* `CUDA_HOME=$CONDA_PREFIX`, `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`
  (needed by the host C++ compiler for `cuda_runtime.h`), conda GCC 14
  `CC/CXX`, `TORCH_CUDA_ARCH_LIST=8.9`, `MAX_JOBS=2` (override for faster builds).
* Never changes `TORCH_EXTENSIONS_DIR`: the gsplat extension stays cached in
  `~/.cache/torch_extensions/py311_cu128/gsplat_cuda/`.

### First gsplat use

The first `import gsplat.cuda._backend` after a source-path change JIT-compiles
the extension (measured: 252 s with `MAX_JOBS=8`; warm import afterwards
2.5 s). This is compilation, not a hang — `VERBOSE=1` shows ninja progress.
Changing the compiler, source paths or flags forces a rebuild; environment
variables like `CPATH`/`MAX_JOBS` do not.

## Verify

```bash
python -m parttwin1d.cli.doctor --provenance --check-gsplat   # verdict: COMPATIBLE, runtime mode: source
python -m pytest -q                                           # unit tests; GPU/GUI tests SKIPPED unless opted in
```

## Remote use (SSH)

The GUI binds `127.0.0.1` by default. From your laptop:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@workstation
# then open http://127.0.0.1:8080 locally
```

If the port is busy viser binds the next free one; the CLI prints a warning and
the actual URL.
