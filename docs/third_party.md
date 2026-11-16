# Third-party sources and runtime dependencies

Runtime mode: **`source`** — `mani_skill`, `gsplat` and `nerfview` are imported from the
pinned Git submodules below via editable `--no-deps` installs. `parttwin doctor
--provenance` verifies this on every run (it prints each `module.__file__`, the
submodule `HEAD`, the SHA recorded in the superproject, and fails on mismatch).

## Git submodules (`third_party/`, pinned in the superproject)

| Path | Upstream URL | Pinned commit (full SHA) | Tag / version | License | Purpose | Runtime import |
|---|---|---|---|---|---|---|
| `third_party/ManiSkill` | https://github.com/mani-skill/ManiSkill.git | `a4a4f9272ad64b1564035874b605ceb687b63ed8` | `v3.0.1` / `mani_skill==3.0.1` | Apache-2.0 (`LICENSE`) | `BaseEnv`, URDF loader, PhysX articulation structs | `import mani_skill` → `third_party/ManiSkill/mani_skill/` |
| `third_party/gsplat` | https://github.com/nerfstudio-project/gsplat.git | `937e29912570c372bed6747a5c9bf85fed877bae` | `v1.5.3` / `gsplat==1.5.3` | Apache-2.0 (`LICENSE`) | CUDA Gaussian rasterizer + SH kernel (JIT-compiled) | `import gsplat` → `third_party/gsplat/gsplat/`; extension `~/.cache/torch_extensions/py311_cu128/gsplat_cuda/gsplat_cuda.so` |
| `third_party/gsplat/gsplat/cuda/csrc/third_party/glm` (nested) | https://github.com/g-truc/glm.git | `2d4c4b4dd31fde06cfffad7915c2b3006402322f` | (pinned by gsplat v1.5.3) | MIT / Happy Bunny | header-only math for gsplat CUDA | compiled into `gsplat_cuda.so` |
| `third_party/nerfview` | https://github.com/nerfstudio-project/nerfview.git | `4538024fe0d15fd1a0e4d760f3695fc44ca72787` | `v0.1.2-3-g4538024` / `nerfview==0.1.2` | **MIT declared in `pyproject.toml`**; no LICENSE file at this commit (upstream added an Apache-2.0 `LICENSE` later in `e6d56fd`) — flag for legal review before redistribution | `Viewer`, `CameraState`, per-client render thread, `rerender()` | `import nerfview` → `third_party/nerfview/nerfview/` |

### Why nerfview `4538024` (not upstream `HEAD` `bd802d7`)

* It is the exact commit gsplat v1.5.3 pins in `examples/requirements.txt`
  (`git+https://github.com/nerfstudio-project/nerfview@4538024…`), i.e. the
  version gsplat's own `simple_viewer.py` / `gsplat_viewer.py` were written for.
* `HEAD` removes adaptive low-resolution rendering during camera motion and the
  `max(dt, 1e-10)` guard in the FPS estimate (a zero division there would hit
  nerfview's `os._exit(1)` error path).
* API verified at this commit: `render_fn(camera_state, render_tab_state)`;
  `CameraState(fov, aspect, c2w)` with OpenCV c2w from viser; `get_K((W,H))`
  gives `fx = fy = H/2/tan(fov/2)`, centred principal point;
  `Viewer.rerender(_)`; renderer threads catch `InterruptRenderException`,
  any other exception → `os._exit(1)` (our callback catches everything else).

### Installed wheel vs submodule source (diagnosed before switching)

* `gsplat 1.5.3` wheel: every `.py/.cu/.cpp/.h` file byte-identical to the
  `v1.5.3` tag (`diff -rq`), so no API difference. Switching to the submodule
  changes the absolute source paths in the ninja build → one full JIT rebuild
  (252 s, then cached; warm import 2.5 s).
* `mani_skill 3.0.1` wheel: all Python modules identical to `v3.0.1`; the source
  tree only adds non-packaged extras (READMEs, `shaders/`, example assets).

## Runtime dependencies (not submodules)

| Package | Version | License | Source | Notes |
|---|---|---|---|---|
| torch | 2.11.0+cu128 | BSD-3-Clause | existing conda env wheel | not reinstalled |
| sapien | 3.0.3 | MIT | binary wheel | no source submodule (would be unused) |
| viser | 1.1.1 | MIT | pip | browser frontend + websocket server; source not modified, so not a submodule |
| plyfile | 1.1.5 | **GPL-3.0-or-later** | pip | PLY parsing; note copyleft if redistributing |
| splines | 0.3.3 | MIT | pip | imported by `nerfview.render_panel` |
| playwright | 1.65.2 | Apache-2.0 | pip (dev only) | drives system Google Chrome in GUI tests |
| CUDA toolkit (nvcc) | 12.8.93 | NVIDIA EULA | conda `cuda-nvcc` | host headers under `$CONDA_PREFIX/targets/x86_64-linux/include` |

## Reference-only projects (not added)

`ManiSkill-GS`, `GSWorld`, `GaussianRenderer` were not needed; none is
installed and none of their dependency stacks touched the environment.
