"""Repository hygiene: single production namespace, no stale smoke packages."""
import importlib
import importlib.metadata as md
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parttwin1d_from_src():
    import parttwin1d

    assert Path(parttwin1d.__file__).resolve().is_relative_to(ROOT / "src")


def test_no_smoke_packages_or_trash_imports():
    assert importlib.util.find_spec("parttwin_smoke") is None
    for dist in ("parttwin-maniskill-gs-smoke",):
        try:
            md.version(dist)
            raise AssertionError(f"stale distribution {dist} still installed")
        except md.PackageNotFoundError:
            pass
    assert not [p for p in sys.path if ".local/share/Trash" in p]
    for d in md.distributions():
        du = d.read_text("direct_url.json") or ""
        assert ".local/share/Trash" not in du, d.metadata["Name"]


def test_light_modules_import_without_sapien_or_cuda_init():
    code = ("import sys, parttwin1d.config, parttwin1d.geometry.se3, parttwin1d.assets.gaussian_ply, "
            "parttwin1d.sim.env, parttwin1d.rendering.gsplat_backend; "
            "assert 'sapien' not in sys.modules and 'mani_skill' not in sys.modules, 'eager simulator import'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_root_pytest_collects_without_errors():
    """`python -m pytest` from the repo root must collect cleanly (no legacy, no ROS plugins)."""
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "legacy/" not in r.stdout
    assert "error" not in r.stdout.strip().splitlines()[-1].lower(), r.stdout[-2000:]


def test_no_legacy_package_imports_in_src():
    import re

    pat = re.compile(r"^\s*(import|from)\s+(parttwin_smoke|legacy)\b", re.M)
    for p in (ROOT / "src").rglob("*.py"):
        assert not pat.search(p.read_text()), p
