import sys
import types
import numpy as np
import torch
from pathlib import Path
from parttwin1d.sim.assets import AssetConfig, GaussianLink, CameraSpec
from parttwin1d.sim.renderer import ArticulatedGaussianRenderer
from parttwin1d.sim.geometry import matrix_from_pose


def test_cpu_mock_rasterizer_keeps_intrinsics_and_link_poses(monkeypatch):
    import parttwin1d.sim.renderer as render_module
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def fake_load(path,device,limit):
        return {'means':torch.tensor([[0.2,0.0,0.3]]),
                'quats':torch.tensor([[1.,0.,0.,0.]]),
                'scales':torch.tensor([[0.01,0.01,0.01]]),
                'opacities':torch.ones((1,)),
                'sh':torch.tensor([[[0.5,0.25,0.75]]]),
                'degree':0,'count':1}
    monkeypatch.setattr(render_module,'load_graphdeco_ply',fake_load)
    called={}
    def rasterization(**kw):
        called.update(kw)
        return (torch.zeros((1,4,5,4)), torch.ones((1,4,5,1)), {})
    gsplat=types.ModuleType('gsplat')
    gsplat_rendering=types.ModuleType('gsplat.rendering')
    gsplat_cuda=types.ModuleType('gsplat.cuda')
    gsplat_wrap=types.ModuleType('gsplat.cuda._wrapper')
    gsplat_rendering.rasterization=rasterization
    gsplat_wrap.spherical_harmonics=lambda d,dirs,coeffs:torch.zeros((len(dirs),3))
    monkeypatch.setitem(sys.modules,'gsplat',gsplat)
    monkeypatch.setitem(sys.modules,'gsplat.rendering',gsplat_rendering)
    monkeypatch.setitem(sys.modules,'gsplat.cuda',gsplat_cuda)
    monkeypatch.setitem(sys.modules,'gsplat.cuda._wrapper',gsplat_wrap)
    camera=CameraSpec('test',5,4,np.array([[200.,0,1.5],[0,240.,2.5],[0,0,1]]),np.eye(4))
    spec=GaussianLink('moving',Path('ignored.ply'),np.eye(4))
    config=AssetConfig(Path('x.json'),Path('fake.urdf'),(spec,),(camera,),True,np.eye(4),(1,1,1),-1)
    renderer=ArticulatedGaussianRenderer(config,device='cpu')
    pose=types.SimpleNamespace(p=torch.tensor([[1.,2.,3.]]),q=torch.tensor([[1.,0.,0.,0.]]))
    art=types.SimpleNamespace(links_map={'moving':types.SimpleNamespace(pose=pose)})
    d=renderer.render_tensor(art,camera)
    assert d['rgb'].shape==(4,5,3)
    torch.testing.assert_close(called['Ks'][0],torch.tensor(camera.K,dtype=torch.float32))
    torch.testing.assert_close(called['means'][0],torch.tensor([1.2,2.,3.3]))
    np.testing.assert_allclose(d['T_world_ply']['moving'][:3,3],[1,2,3])
