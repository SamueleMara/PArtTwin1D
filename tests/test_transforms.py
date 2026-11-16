import numpy as np
import pytest
import torch
from parttwin_smoke.transforms import (validate_se3,compose,inverse,transform_points,
                                        pose_wxyz_to_matrix,quat_mul_wxyz,quat_rotate_wxyz,
                                        view_from_eye_target)
from parttwin_smoke.assets import decode_rest_sh,load_manifest


def test_inverse_composition():
    T=pose_wxyz_to_matrix([1.,2.,3.],[np.sqrt(.5),0,0,np.sqrt(.5)])
    np.testing.assert_allclose(compose(T,inverse(T)),np.eye(4),atol=1e-12)


def test_hinge_xyz_and_covariance():
    q=np.sqrt(.5)
    T=pose_wxyz_to_matrix([0,0,0],[q,0,0,q])
    pt=transform_points(T,[[.5,0,.3]])
    np.testing.assert_allclose(pt,[[0,.5,.3]],atol=1e-12)
    cov=np.diag([.09,.002,.001])
    c=T[:3,:3]@cov@T[:3,:3].T
    np.testing.assert_allclose(np.linalg.eigvalsh(c),np.linalg.eigvalsh(cov),atol=1e-12)


def test_wxyz_rotations():
    q=torch.tensor([np.sqrt(.5),0.,0.,np.sqrt(.5)],dtype=torch.float64)
    v=torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    out=quat_rotate_wxyz(q[None],v)
    torch.testing.assert_close(out,torch.tensor([[0.,1.,0.]],dtype=torch.float64),atol=1e-12,rtol=0)
    qq=quat_mul_wxyz(q[None],q[None]);torch.testing.assert_close(qq,torch.tensor([[0.,0.,0.,1.]],dtype=torch.float64),atol=1e-12,rtol=0)


def test_camera_optical_forward():
    v=view_from_eye_target([1,0,0],[0,0,0])
    p=transform_points(v,[[0.,0.,0.]])[0]
    np.testing.assert_allclose(p[:2],[0,0],atol=1e-12)
    assert p[2] > 0


def test_invalid_scale_and_reflection():
    T=np.eye(4);T[0,0]=2
    with pytest.raises(ValueError):validate_se3(T)
    T=np.eye(4);T[0,0]=-1
    with pytest.raises(ValueError):validate_se3(T)


def test_graphdeco_sh_channel_major_layout():
    # Graphdeco lays f_rest as all R, then all G, then all B.
    packed=np.array([[1,2, 10,20, 100,200]],dtype=np.float32)
    decoded=decode_rest_sh(packed)
    assert decoded.shape==(1,2,3)
    np.testing.assert_allclose(decoded[0],[[1,10,100],[2,20,200]])


def test_manifest(tmp_path):
    (tmp_path/'a.urdf').write_text('<robot name="a"></robot>')
    (tmp_path/'a.ply').write_text('placeholder')
    (tmp_path/'manifest.json').write_text('{"urdf":"a.urdf","links":{"door":{"ply":"a.ply"}}}')
    u,s,_=load_manifest(tmp_path/'manifest.json')
    assert u.name=='a.urdf' and s[0].link=='door'
    np.testing.assert_allclose(s[0].T_link_ply,np.eye(4))
