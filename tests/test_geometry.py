import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from parttwin1d.sim.geometry import (se3, invert, matrix_from_pose, quat_from_matrix,
                                    optical_view_from_look_at, transform_points, quat_multiply)


def test_inverse():
    T=matrix_from_pose([1,2,3],[0.9,0.1,0.1,0.2])
    np.testing.assert_allclose(invert(T)@T,np.eye(4),atol=1e-7)


def test_wxyz_scalar_first():
    q=quat_from_matrix(Rotation.from_euler('z',90,degrees=True).as_matrix())
    np.testing.assert_allclose(abs(q[0]),np.cos(np.pi/4),atol=1e-8)
    R=matrix_from_pose([0,0,0],q)[:3,:3]
    np.testing.assert_allclose(R @ [1,0,0],[0,1,0],atol=1e-8)


def test_optical_camera_sign():
    eye=np.array([0.,0.,0.]); target=np.array([0.,0.,1.])
    T=optical_view_from_look_at(eye,target,up=(0,1,0))
    point_camera=transform_points(T,np.array([[0,0,2.]]))[0]
    np.testing.assert_allclose(point_camera,[0,0,2])


def test_se3_rejects_scale():
    T=np.diag([2.,1.,1.,1.])
    with pytest.raises(ValueError,match='unscaled'):
        se3(T)


def test_hamilton_product():
    import torch
    q=torch.tensor([[0.70710678,0.,0.,0.70710678]])
    identity=torch.tensor([[1.,0.,0.,0.]])
    torch.testing.assert_close(quat_multiply(identity,q),q)
    torch.testing.assert_close(quat_multiply(q,q),torch.tensor([[0.,0.,0.,1.]]),atol=1e-5,rtol=0.)
