# Verified environment lock — `parttwin_sim` (2026-11-16)

Workstation: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9, 8 GB), driver 580.178.04,
Ubuntu 24.04, Python 3.11.18 (conda env `parttwin_sim`).

## Toolchain

| Item | Version / path |
|---|---|
| CUDA toolkit (conda `cuda-nvcc`, `cuda-version`) | 12.8.93 / 12.8 — `$CONDA_PREFIX/bin/nvcc` |
| Host compiler for gsplat JIT | `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++` (GCC 14, set by `scripts/activate_sim.sh`) |
| `CUDA_HOME` | `$CONDA_PREFIX` |
| `CPATH` (required) | `$CONDA_PREFIX/targets/x86_64-linux/include` — see below |
| `TORCH_CUDA_ARCH_LIST` | `8.9` |
| gsplat extension cache | `~/.cache/torch_extensions/py311_cu128/gsplat_cuda/gsplat_cuda.so` |

`CPATH` is required because conda's CUDA 12.8 places `cuda_runtime.h` under
`targets/x86_64-linux/include`; nvcc finds it through `nvcc.profile`, but the
*host* C++ compiler (gsplat's `.cpp` bindings) only gets `-isystem $CUDA_HOME/include`
from PyTorch → `fatal error: cuda_runtime.h: No such file or directory`.

## Python packages (exact, from `pip freeze`)

| Package | Version | How installed |
|---|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | pre-existing (untouched) |
| sapien | 3.0.3 | pre-existing (untouched) |
| mani_skill | 3.0.1 | `-e third_party/ManiSkill` @ `a4a4f9272ad64b1564035874b605ceb687b63ed8` (`--no-deps`) |
| gsplat | 1.5.3 | `-e third_party/gsplat` @ `937e29912570c372bed6747a5c9bf85fed877bae` (`BUILD_NO_CUDA=1 --no-deps`) |
| nerfview | 0.1.2 | `-e third_party/nerfview` @ `4538024fe0d15fd1a0e4d760f3695fc44ca72787` (`--no-deps`) |
| parttwin1d | 0.4.0 | `-e .` (`--no-deps`) |
| viser | 1.1.1 | pip (added) |
| splines | 0.3.3 | pip (added) |
| msgspec / websockets / zstandard | 0.22.0 / 16.1.1 / 0.25.0 | pip (viser deps, added) |
| rich | 14.3.4 | pip (viser requires `<15`; was 15.0.0) |
| playwright / greenlet | 1.65.2 / 3.5.6 | pip (dev, added) |
| numpy | 2.4.6 | pre-existing |
| scipy | 1.17.1 | pre-existing |
| plyfile | 1.1.5 | pre-existing |
| PyYAML | 6.0.3 | pre-existing |
| pillow | 13.0.0 | pre-existing |
| matplotlib | 3.11.2 | pre-existing |
| jaxtyping | 0.3.11 | pre-existing |
| pytest | 9.2.0 | pre-existing |

Removed: `parttwin-maniskill-gs-smoke 0.3.0` (stale editable install mapping
`parttwin_smoke` to a non-existent path). Full before/after freezes:
`outputs/logs/pip_freeze_before_refactor.txt`, `outputs/logs/pip_freeze_after_refactor.txt`
(gitignored run artefacts). `python -m pip check`: no broken requirements.

## Reproduce on this machine

```bash
conda activate parttwin_sim
cd ~/PArtTwin1D
bash scripts/bootstrap_submodules.sh     # idempotent; first gsplat JIT ≈ 4 min
```
