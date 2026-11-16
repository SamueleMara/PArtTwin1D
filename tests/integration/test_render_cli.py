"""End-to-end offline render CLI (real PhysX + real gsplat) on both PhysX backends."""
import json
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import HINGED, ROOT, requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu, pytest.mark.physx]


@pytest.mark.parametrize("backend", ["physx_cpu", "physx_cuda"])
def test_render_cli_sweep(tmp_path, backend):
    from PIL import Image

    out = tmp_path / backend
    r = subprocess.run([sys.executable, "-m", "parttwin1d.cli.render", "--config", str(HINGED), "--joint", "door_hinge",
                        "--q-start", "0", "--q-end", "1.3", "--frames", "6", "--out", str(out), "--sim-backend", backend,
                        "--segmentation"], capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads((out / "report.json").read_text())
    ch = rep["checks"]
    assert ch["moving_link_max_rotation_err_rad"] < 1e-4
    assert ch["moving_link_max_translation_err_m"] < 1e-5
    assert ch["static_base_link_max_translation_m"] < 1e-6 and ch["static_base_link_max_rotation_rad"] < 1e-6
    assert [round(s["q"]["door_hinge"], 4) for s in rep["sequence"]] == [round(x, 4) for x in np.linspace(0, 1.3, 6)]
    for cam in ("front_oblique", "left_high"):
        assert rep["cameras"][cam]["first_last_mean_abs_rgb_change_0_255"] > 2.0
        img = np.asarray(Image.open(out / cam / "0005_rgb.png"))
        assert img.shape == ((480, 640, 3) if cam == "front_oblique" else (384, 512, 3))
        md = json.loads((out / cam / "0005_meta.json").read_text())
        assert md["q"]["door_hinge"] == pytest.approx(1.3, abs=1e-5)
        assert (out / f"{cam}.gif").stat().st_size > 1000
    (ROOT / "outputs" / "test_artifacts").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "test_artifacts" / f"render_cli_report_{backend}.json").write_text(
        json.dumps({k: rep[k] for k in ("checks", "first_render_ms", "timing_s", "gaussians")}, indent=2))
