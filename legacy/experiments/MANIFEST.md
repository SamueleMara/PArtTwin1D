# legacy/experiments — archived, NOT production code

Nothing in this directory is importable as part of `parttwin1d`, is on
`PYTHONPATH`, or is collected by pytest (`norecursedirs` in `pyproject.toml`).
It is kept only for provenance of the pre-refactor smoke iterations.

| Path | Origin | Why archived | Replacement |
|---|---|---|---|
| `tests/test_sh_directions.py` | root `tests/` | Imports removed package `parttwin_smoke` → root `pytest` collection error | `tests/unit/test_splat_link_pose.py`, `tests/integration/test_gsplat_render.py::test_sh_rigid_invariance` |
| `tests/test_transforms.py` | root `tests/` | Imports `parttwin_smoke` → collection error | `tests/unit/test_se3.py`, `tests/unit/test_ply.py` |
| `tests/test_robotless_env.py` | root `tests/` | Imports `parttwin_smoke`; mocked `BaseEnv` | `tests/integration/test_maniskill_robotless.py` (real ManiSkill 3.0.1) |
| `tests/test_maniskill_no_robot.py` | root `tests/` | Fully mocked ManiSkill/SAPIEN modules | `tests/integration/test_maniskill_robotless.py` |
| `tests/test_renderer_contract.py` | root `tests/` | Mocked gsplat rasterizer | `tests/integration/test_gsplat_render.py` (real CUDA kernel) |
| `scripts/run_synthetic.sh`, `scripts/setup_env.sh`, `scripts/requirements-smoke.txt` | repo root | Drive `python -m parttwin_smoke.*` | `Makefile`, `scripts/*.sh`, `requirements/` |
| `docs/INSTALL_UBUNTU.md`, `docs/README_MANISKILL.md` | repo root | Describe v2/v3 smoke ZIP installs into `parttwin_maniskill_gs_v3/` | `docs/installation.md`, `docs/architecture.md` |
| `fixtures/synthetic_fixture/`, `fixtures/create_synthetic_fixture.py` | `examples/` | Duplicate of the hinged-box fixture for the smoke package | `examples/hinged_box/` |
| `fixtures/hinged_box.v0.json`, `fixtures/real_object.template.v0.json` | `configs/` | Pre-schema JSON configs (implicit identity `T_link_ply`, no `schema_version`) | `configs/examples/hinged_box.yaml`, `configs/real_object.template.yaml` |

## Quarantined untracked files (`quarantine/`, gitignored content)

| File | SHA-256 | Notes |
|---|---|---|
| `quarantine/e .zip` | see `quarantine/SHA256SUMS` | Untracked 28 KB archive found at repo root during the audit (2026-11-16). Contains a copy of the pre-refactor `parttwin1d/sim`, `scripts/`, `tests/` (21 files). Not deleted because it was user-created and untracked. Safe to delete once confirmed redundant. |

## Stale environment state removed during the audit

* Editable distribution `parttwin-maniskill-gs-smoke 0.3.0` in `parttwin_sim`
  (`__editable___parttwin_maniskill_gs_smoke_0_3_0_finder.py` mapping
  `parttwin_smoke -> /home/samuelemara/PArtTwin1D/parttwin_smoke`, a path that
  no longer existed). Uninstalled with `pip uninstall parttwin-maniskill-gs-smoke`.
* Earlier iterations of that package live in `~/.local/share/Trash/files/`
  (`parttwin_smoke`, `parttwin_maniskill_gs_v2`, `PArtTwin1D_ManiSkill_3DGS_v3_Robotless_Fix`).
  Nothing in `parttwin_sim` resolves there anymore; `tests/unit/test_imports.py`
  asserts this at every run.
