"""Environment and source-provenance doctor.

    python -m parttwin1d.cli.doctor --provenance [--check-gsplat] [--json out.json]

Reports versions, absolute import paths, submodule SHAs vs pins, CUDA_HOME,
nvcc, torch CUDA, GPU capability and the runtime mode:

* ``source``          mani_skill / gsplat / nerfview import from ``third_party/`` (target).
* ``wheel-fallback``  at least one comes from site-packages (must be diagnosed).

Exit code 1 on any incompatibility (wrong version/SHA, stale smoke package,
imports from the Trash, CUDA unavailable, ...).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .._pins import CUDA_TOOLKIT, RUNTIME, SUBMODULES

REPO = Path(__file__).resolve().parents[3]


def _git(*args, cwd=REPO) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<git error {exc}>"


def _version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def collect(check_gsplat: bool = False) -> dict:
    rep = {"python": sys.version.split()[0], "executable": sys.executable, "repo": str(REPO),
           "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "problems": [], "warnings": []}
    P, W = rep["problems"], rep["warnings"]

    import parttwin1d

    rep["parttwin1d"] = {"file": parttwin1d.__file__, "version": getattr(parttwin1d, "__version__", "?")}
    if not str(Path(parttwin1d.__file__).resolve()).startswith(str(REPO / "src")):
        P.append(f"parttwin1d imported from {parttwin1d.__file__}, not {REPO / 'src'}")

    # stale smoke-era packages / Trash imports
    if importlib.util.find_spec("parttwin_smoke") is not None:
        P.append(f"stale package parttwin_smoke importable: {importlib.util.find_spec('parttwin_smoke').origin}")
    for dist in ("parttwin-maniskill-gs-smoke",):
        if _version(dist):
            P.append(f"stale distribution {dist} installed")
    trash = [p for p in sys.path if "/.local/share/Trash" in p]
    if trash:
        P.append(f"sys.path entries in Trash: {trash}")
    if os.environ.get("PYTHONPATH", "") and "/opt/ros" in os.environ.get("PYTHONPATH", ""):
        W.append("ROS paths in PYTHONPATH; source scripts/activate_sim.sh (it unsets PYTHONPATH)")

    # submodules
    tree = {line.split()[-1]: line.split()[2] for line in _git("ls-tree", "HEAD", "third_party/").splitlines() if line}
    rep["submodules"] = {}
    sources = []
    for sub, (mod, ver, sha) in SUBMODULES.items():
        path = REPO / "third_party" / sub
        head = _git("rev-parse", "HEAD", cwd=path) if path.exists() else None
        dirty = bool(_git("status", "--porcelain", cwd=path)) if path.exists() else None
        try:
            m = importlib.import_module(mod)
            mfile = str(Path(m.__file__).resolve())
        except Exception as exc:
            mfile = f"<import failed: {exc}>"
        from_src = mfile.startswith(str(path.resolve()))
        sources.append(from_src)
        dv = _version(mod) or _version(mod.replace("_", "-"))
        info = {"path": str(path), "head": head, "recorded_in_superproject": tree.get(f"third_party/{sub}"),
                "pinned": sha, "dirty": dirty, "module": mod, "module_file": mfile, "dist_version": dv,
                "expected_version": ver, "imported_from_submodule": from_src}
        rep["submodules"][sub] = info
        if head != sha:
            P.append(f"third_party/{sub} HEAD {head} != pinned {sha}")
        if tree.get(f"third_party/{sub}") not in (None, sha):
            P.append(f"superproject records {tree.get(f'third_party/{sub}')} for {sub}, pin is {sha}")
        if dv != ver:
            P.append(f"{mod} version {dv} != expected {ver}")
        if dirty:
            W.append(f"third_party/{sub} has local modifications")
    rep["runtime_mode"] = "source" if all(sources) else "wheel-fallback"
    if rep["runtime_mode"] != "source":
        W.append("runtime mode is wheel-fallback: some upstream packages are not imported from third_party/")

    rep["runtime"] = {}
    for dist, ver in RUNTIME.items():
        v = _version(dist)
        mod = {"torch": "torch", "sapien": "sapien", "viser": "viser", "plyfile": "plyfile",
               "scipy": "scipy", "numpy": "numpy"}[dist]
        spec = importlib.util.find_spec(mod)
        rep["runtime"][dist] = {"version": v, "expected": ver, "file": spec.origin if spec else None}
        if v != ver:
            (P if dist in ("torch", "sapien") else W).append(f"{dist} {v} != verified {ver}")

    # CUDA
    rep["CUDA_HOME"] = os.environ.get("CUDA_HOME")
    rep["CPATH"] = os.environ.get("CPATH")
    nvcc = shutil.which("nvcc")
    rep["nvcc"] = nvcc
    rep["nvcc_version"] = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:] \
        if nvcc else None
    if not nvcc:
        P.append("nvcc not on PATH (gsplat JIT needs it); source scripts/activate_sim.sh")
    elif f"release {CUDA_TOOLKIT}" not in " ".join(rep["nvcc_version"] or []):
        W.append(f"nvcc is not CUDA {CUDA_TOOLKIT}")
    import torch

    rep["torch_cuda"] = {"available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        rep["torch_cuda"].update({"device": torch.cuda.get_device_name(0), "capability": f"{cap[0]}.{cap[1]}",
                                  "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)})
        arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch and arch != f"{cap[0]}.{cap[1]}":
            W.append(f"TORCH_CUDA_ARCH_LIST={arch} but device is SM {cap[0]}.{cap[1]}")
    else:
        P.append("torch.cuda.is_available() is False")
    if torch.version.cuda and not torch.version.cuda.startswith(CUDA_TOOLKIT):
        W.append(f"torch built for CUDA {torch.version.cuda}, toolkit pin {CUDA_TOOLKIT}")

    from torch.utils.cpp_extension import _get_build_directory

    bd = Path(_get_build_directory("gsplat_cuda", verbose=False))
    rep["gsplat_extension_cache"] = {"dir": str(bd), "so_exists": (bd / "gsplat_cuda.so").exists()}
    if check_gsplat:
        from ..rendering.gsplat_backend import gsplat_cuda_available

        ok, info = gsplat_cuda_available()
        rep["gsplat_extension_cache"]["loaded"] = ok
        rep["gsplat_extension_cache"]["module"] = info
        if not ok:
            P.append(f"gsplat CUDA extension unavailable: {info}")
    rep["verdict"] = "COMPATIBLE" if not P else "INCOMPATIBLE"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="parttwin doctor", description=__doc__.splitlines()[0])
    ap.add_argument("--provenance", action="store_true", help="full provenance report (default)")
    ap.add_argument("--check-gsplat", action="store_true", help="also load the gsplat CUDA extension (may JIT-compile)")
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    args = ap.parse_args(argv)
    rep = collect(check_gsplat=args.check_gsplat)
    print(f"python      {rep['python']}  ({rep['executable']}), conda env {rep['conda_env']}")
    print(f"parttwin1d  {rep['parttwin1d']['file']}")
    print(f"runtime mode: {rep['runtime_mode']}")
    for sub, s in rep["submodules"].items():
        print(f"  third_party/{sub:9s} HEAD {s['head']}  pin {'OK' if s['head'] == s['pinned'] else 'MISMATCH'}"
              f"  {s['module']}=={s['dist_version']}  <- {s['module_file']}")
    for dist, r in rep["runtime"].items():
        print(f"  {dist:8s} {r['version']:<14} (verified {r['expected']})  {r['file']}")
    print(f"CUDA_HOME   {rep['CUDA_HOME']}\nCPATH       {rep['CPATH']}\nnvcc        {rep['nvcc']} {rep['nvcc_version']}")
    print(f"torch CUDA  {rep['torch_cuda']}")
    print(f"gsplat ext  {rep['gsplat_extension_cache']}")
    for w in rep["warnings"]:
        print(f"WARNING: {w}")
    for p in rep["problems"]:
        print(f"PROBLEM: {p}")
    print(f"verdict: {rep['verdict']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2, default=str))
    return 0 if rep["verdict"] == "COMPATIBLE" else 1


if __name__ == "__main__":
    sys.exit(main())
