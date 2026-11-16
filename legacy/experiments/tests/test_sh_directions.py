"""Regression for real gsplat 1.5.3 spherical_harmonics API shape/direction."""
from pathlib import Path
import sys
import types
import numpy as np
import torch
from parttwin_smoke.assets import PartSpec
from parttwin_smoke.renderer import ArticulatedGsRenderer
from parttwin_smoke.transforms import view_from_eye_target

class Pose:
    def __init__(self, p, q):
        self.p = torch.tensor([p], dtype=torch.float32)
        self.q = torch.tensor([q], dtype=torch.float32)
class Link:
    def __init__(self, p, q):
        self.pose = Pose(p, q)
class Art:
    def __init__(self, quat):
        self.links_map = {"door": Link([0,0,0], quat)}

def test_view_dependent_sh_local_directions_follow_joint(monkeypatch):
    calls = []
    def fake_sh(degree, dirs, coeffs):
        # gsplat 1.5.3: dirs [N,3], coeffs [N,K,3]; old code passed a view matrix.
        assert degree == 1
        assert dirs.shape == (2,3)
        assert coeffs.shape == (2,4,3)
        calls.append(dirs.clone())
        return dirs.abs() * 0.05
    def fake_rasterization(**kwargs):
        assert kwargs['colors'].shape == (2,3)
        h,w=kwargs['height'],kwargs['width']
        rgbd=torch.zeros((1,h,w,4),dtype=torch.float32)
        alpha=torch.zeros((1,h,w,1),dtype=torch.float32)
        alpha[0, h//2, w//2, 0] = 1
        return rgbd,alpha,{}
    gsplat=types.ModuleType('gsplat');gsplat.__path__=[]
    gr=types.ModuleType('gsplat.rendering');gr.rasterization=fake_rasterization
    gc=types.ModuleType('gsplat.cuda');gc.__path__=[]
    gw=types.ModuleType('gsplat.cuda._wrapper');gw.spherical_harmonics=fake_sh
    for n,v in [('gsplat',gsplat),('gsplat.rendering',gr),('gsplat.cuda',gc),('gsplat.cuda._wrapper',gw)]:
        monkeypatch.setitem(sys.modules,n,v)
    model=ArticulatedGsRenderer.__new__(ArticulatedGsRenderer)
    model.device=torch.device('cpu')
    means=torch.tensor([[0.3,0.0,0.4],[0.7,0.0,0.4]])
    model.parts=[dict(spec=PartSpec('door',Path('unused'),np.eye(4)), degree=1,
                      dat=dict(means=means, quats=torch.tensor([[1.,0.,0.,0.]]*2),
                               scales=torch.ones((2,3))*.01,
                               opacities=torch.ones(2)*.9,
                               sh=torch.ones((2,4,3))*.1, degree=1,n=2))]
    eye=[1.0,-2.0,1.0]
    v=view_from_eye_target(eye,[0,0,0.3]);K=np.array([[200,0,64],[0,200,64],[0,0,1]])
    model.render(Art([1,0,0,0]),v,K,128,128)
    np.testing.assert_allclose(calls[-1].numpy(),
                               torch.nn.functional.normalize(means-torch.tensor(eye),dim=-1).numpy(),atol=1e-6)
    a=2**-.5
    model.render(Art([a,0,0,a]),v,K,128,128)
    assert not torch.allclose(calls[0],calls[1])
    # The splat geometry is rigidly rotated but SH evaluated in the canonical PLY frame.
