from pathlib import Path
import numpy as np
import pytest
from parttwin1d.sim.assets import load_config, ordered_columns, unpack_graphdeco_sh

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module', autouse=True)
def ensure_demo_gaussians():
    # Generated PLYs are deliberately not committed to the main repository.
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT/'scripts/create_demo_assets.py'),
                    '--out', str(ROOT/'examples/hinged_box')], check=True,
                   capture_output=True, text=True)


def test_real_example_manifest():
    c=load_config(ROOT/'configs/hinged_box.json')
    assert len(c.links)==2
    assert c.cameras[0].K[0,0] == 590
    assert c.links[1].link=='door_link'


def test_graphdeco_sh_channel_order():
    dc=np.array([[1,2,3]],np.float32)
    rest=np.array([[11,12,21,22,31,32]],np.float32)
    out=unpack_graphdeco_sh(dc,rest)
    assert out.shape==(1,3,3)
    np.testing.assert_array_equal(out[0,1],[11,21,31])
    np.testing.assert_array_equal(out[0,2],[12,22,32])


def test_missing_sh_index_detected():
    with pytest.raises(ValueError, match='continuously'):
        ordered_columns(('f_rest_0','f_rest_2'),'f_rest_')


def test_camera_intrinsics_preserved():
    c=load_config(ROOT/'configs/hinged_box.json')
    assert c.cameras[0].K[0,2]==320
    assert c.cameras[0].K[1,2]==240
