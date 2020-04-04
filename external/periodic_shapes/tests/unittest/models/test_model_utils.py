import math
import torch
import numpy as np
from models import model_utils
from external.QuaterNet.common import quaternion
import time

def test_deterministic_m_vector():
    """Test m_vector has max value at specified by m."""
    m = 5
    max_m = m + 1
    target_m_vector = torch.eye(max_m).view(1, 1, max_m, max_m)[:, :,
                                                                m, :].float().unsqueeze(-1)
    logits = target_m_vector * 10 + 1

    m_vector = model_utils.get_deterministic_m_vector(logits)

    torch.all(torch.eq(m_vector, target_m_vector))


def test_get_2d_rotation_matrix():
    """Test if rotation matrix shape and values are as expected."""
    n = 2
    theta = 0.2
    rotation = torch.tensor([theta] * n).view(1, -1)
    target_rotation_matrix = torch.tensor(
        [math.cos(theta), -math.sin(theta),
         math.sin(theta),
         math.cos(theta)]).view(1, 1, 4).repeat(1, n, 1)
    rotation_matrix = model_utils.get_2d_rotation_matrix(rotation)
    torch.all(torch.eq(target_rotation_matrix, rotation_matrix))
    assert [*target_rotation_matrix.shape] == [1, n, 4]

    # Test inverse rotation matrix
    inv_rotation_matrix = model_utils.get_2d_rotation_matrix(rotation,
                                                             inv=True)
    target_inv_rotation_matrix = torch.tensor(
        [math.cos(theta),
         math.sin(theta), -math.sin(theta),
         math.cos(theta)]).view(1, 1, 4).repeat(1, n, 1)
    torch.all(torch.eq(target_inv_rotation_matrix, inv_rotation_matrix))


def test_apply_3d_rotation():
    B = 3
    N = 5
    P = 10000 
    rotation_eu = np.array([0., math.pi / 2., math.pi / 2])
    rotation_qua = quaternion.euler_to_quaternion(rotation_eu, 'xyz')
    rotation_qua_tensor = torch.from_numpy(rotation_qua).view(1, 1, 4).repeat(
        B, N, 1).float()

    point = torch.tensor([0., 1., 0.]).view(1, 1, 1, 3).repeat(B, N, P, 1)
    rotated_point = model_utils.apply_3d_rotation(point, rotation_qua_tensor)

    target_point = torch.tensor([0., 0., 1.]).view(1, 1, 1,
                                                   3).repeat(B, N, P, 1)

    assert torch.allclose(target_point, rotated_point)

    rerotated_point = model_utils.apply_3d_rotation(rotated_point,
                                                    rotation_qua_tensor,
                                                    inv=True)

    assert torch.allclose(point, rerotated_point)
