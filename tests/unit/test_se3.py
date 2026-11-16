"""SE(3) / quaternion contract (CPU, NumPy + Torch)."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from parttwin1d.geometry.se3 import (
    FrameError, axis_angle_matrix, compose, invert, invert_torch, matrix_from_pose, matrix_to_quat_torch,
    quat_from_matrix, quat_multiply, quat_to_matrix_torch, rotate_covariance, transform_points, validate_se3,
)

RNG = np.random.default_rng(0)


def rand_T():
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=RNG.integers(1e9)).as_matrix()
    T[:3, 3] = RNG.normal(size=3)
    return T


def test_inverse_and_compose():
    A, B = rand_T(), rand_T()
    np.testing.assert_allclose(invert(A) @ A, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(compose(A, B), A @ B, atol=1e-12)
    x = RNG.normal(size=(5, 3))
    np.testing.assert_allclose(transform_points(compose(A, B), x), transform_points(A, transform_points(B, x)), atol=1e-12)


def test_point_transform_direction_convention():
    # T_W_L maps link points into world: a door point (0.5,0,0.3) under +90deg yaw -> (0,0.5,0.3)
    T = matrix_from_pose([0, 0, 0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    np.testing.assert_allclose(transform_points(T, [[0.5, 0, 0.3]]), [[0, 0.5, 0.3]], atol=1e-12)


def test_covariance_rotation_matches_quaternion_composition():
    R_wp = Rotation.random(random_state=1).as_matrix()
    q_g = Rotation.random(random_state=2).as_quat()[[3, 0, 1, 2]]
    S = np.diag([0.3, 0.02, 0.01]) ** 2
    R_g = matrix_from_pose([0, 0, 0], q_g)[:3, :3]
    sigma_p = R_g @ S @ R_g.T
    sigma_w_ref = rotate_covariance(R_wp, sigma_p)
    q_w = quat_multiply(torch.tensor(quat_from_matrix(R_wp))[None], torch.tensor(q_g)[None])[0].numpy()
    R_w = matrix_from_pose([0, 0, 0], q_w)[:3, :3]
    np.testing.assert_allclose(R_w @ S @ R_w.T, sigma_w_ref, atol=1e-12)
    np.testing.assert_allclose(R_w, R_wp @ R_g, atol=1e-12)  # R(q_wp ⊗ q_g) = R_wp R_g


def test_scalar_first_quaternion():
    q = quat_from_matrix(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    np.testing.assert_allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-12)
    assert q[0] >= 0


def test_torch_quat_matrix_roundtrip():
    R = torch.tensor(Rotation.random(64, random_state=3).as_matrix())
    q = matrix_to_quat_torch(R)
    assert (q[:, 0] >= 0).all()
    torch.testing.assert_close(quat_to_matrix_torch(q), R, atol=1e-10, rtol=0)
    # 180-degree rotations (trace = -1) are handled
    R180 = torch.tensor(Rotation.from_rotvec([[np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]]).as_matrix())
    torch.testing.assert_close(quat_to_matrix_torch(matrix_to_quat_torch(R180)), R180, atol=1e-10, rtol=0)


def test_torch_inverse():
    T = torch.tensor(np.stack([rand_T() for _ in range(4)]))
    torch.testing.assert_close(invert_torch(T) @ T, torch.eye(4, dtype=T.dtype).expand(4, 4, 4), atol=1e-12, rtol=0)


def test_hamilton_product():
    q = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    torch.testing.assert_close(quat_multiply(torch.tensor([[1.0, 0, 0, 0]]), q), q)
    torch.testing.assert_close(quat_multiply(q, q), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("bad, msg", [
    (np.diag([2.0, 1, 1, 1]), "orthonormal"),
    (np.diag([-1.0, 1, 1, 1]), "reflection"),
    (np.eye(3), "4x4"),
    (np.full((4, 4), np.nan), "non-finite"),
])
def test_validate_rejects(bad, msg):
    with pytest.raises(FrameError, match=msg):
        validate_se3(bad)


def test_bottom_row_rejected():
    T = np.eye(4)
    T[3, 0] = 0.1
    with pytest.raises(FrameError, match="bottom row"):
        validate_se3(T)


def test_axis_angle_about_pivot():
    T = axis_angle_matrix([0, 0, 1], np.pi / 2, pivot=[1, 0, 0])
    np.testing.assert_allclose(transform_points(T, [[1, 0, 0]]), [[1, 0, 0]], atol=1e-12)  # pivot fixed
    np.testing.assert_allclose(transform_points(T, [[2, 0, 0]]), [[1, 1, 0]], atol=1e-12)


def test_zero_quaternion_rejected():
    with pytest.raises(FrameError):
        matrix_from_pose([0, 0, 0], [0, 0, 0, 0])
