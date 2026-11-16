"""Opt-in *real* ManiSkill 3 / SAPIEN / gsplat integration test.

Usage in existing environment:
PARTTWIN_RUN_GPU_TESTS=1 python -m pytest -v tests/test_integration_gpu.py

Tests actual PhysX joint poses and actual CUDA rasterization. Never mocks them.
"""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get('PARTTWIN_RUN_GPU_TESTS') != '1',
                    reason='GPU integration requires explicit opt-in on workstation')
def test_actual_articulation_and_gaussian_rasterizer():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mani_skill')
    pytest.importorskip('sapien')
    pytest.importorskip('gsplat')
    pytest.importorskip('plyfile')
    assert torch.cuda.is_available()
    subprocess.run([sys.executable, str(ROOT/'scripts/create_demo_assets.py'),
                    '--out', str(ROOT/'examples/hinged_box')], check=True)
    from parttwin1d.sim.assets import load_config
    from parttwin1d.sim.env import create_articulated_scene, set_joint_qpos
    from parttwin1d.sim.renderer import ArticulatedGaussianRenderer
    from parttwin1d.sim.geometry import world_from_mani_skill_pose
    cfg = load_config(ROOT/'configs/hinged_box.json')
    env = create_articulated_scene(cfg, 'physx_cpu')
    try:
        env.reset(seed=0)
        art = env.twin
        render = ArticulatedGaussianRenderer(cfg)
        camera = cfg.cameras[0]
        set_joint_qpos(env, 'door_hinge', 0.0)
        base0 = world_from_mani_skill_pose(art.links_map['base_link'].pose)
        door0 = world_from_mani_skill_pose(art.links_map['door_link'].pose)
        image0 = render.render_tensor(art,camera)
        set_joint_qpos(env, 'door_hinge', 1.1)
        base1 = world_from_mani_skill_pose(art.links_map['base_link'].pose)
        door1 = world_from_mani_skill_pose(art.links_map['door_link'].pose)
        image1 = render.render_tensor(art,camera)
        np.testing.assert_allclose(base0,base1,atol=1e-4)
        assert np.linalg.norm(door1[:3,:3]-door0[:3,:3]) > 0.2
        assert image0['rgb'].device.type == 'cuda'
        assert image0['rgb'].shape == (480,640,3)
        assert int((image0['alpha']>0.01).sum().item()) > 200
        assert int((image1['alpha']>0.01).sum().item()) > 200
        assert (image0['rgb']-image1['rgb']).abs().mean().item() > 0.004
    finally:
        env.close()
