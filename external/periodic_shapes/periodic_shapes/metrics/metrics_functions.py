import torch
from kaolin.nnsearch import nnsearch
from kaolin.metrics.point import SidedDistance

def chamfer_distance_l1(S1: torch.Tensor, S2: torch.Tensor,
                     w1: float = 1., w2: float = 1.):
    r"""Computes the chamfer distance between two point clouds

    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            w1: (float): weighting of forward direction
            w2: (float): weighting of backward direction

    Returns:
            torch.Tensor: chamfer distance between two point clouds S1 and S2

    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> chamfer_distance(A,B)
            tensor(0.1868)

    """

    assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
    assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

    dist_to_S2 = directed_distance_l1(S1, S2)
    dist_to_S1 = directed_distance_l1(S2, S1)

    distance = w1 * dist_to_S2 + w2 * dist_to_S1

    return distance


def directed_distance_l1(S1: torch.Tensor, S2: torch.Tensor, mean: bool = True):
    r"""Computes the average distance from point cloud S1 to point cloud S2

    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            mean (bool): if the distances should be reduced to the average

    Returns:
            torch.Tensor: ditance from point cloud S1 to point cloud S2

    Args:

    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> directed_distance(A,B)
            tensor(0.1868)

    """

    if S1.is_cuda and S2.is_cuda:
        sided_minimum_dist = SidedDistance()
        closest_index_in_S2 = sided_minimum_dist(
            S1.unsqueeze(0), S2.unsqueeze(0))[0]
        closest_S2 = torch.index_select(S2, 0, closest_index_in_S2)

    else:
        from time import time
        closest_index_in_S2 = nnsearch(S1, S2)
        closest_S2 = S2[closest_index_in_S2]

    dist_to_S2 = (((S1 - closest_S2).abs()).sum(dim=-1))
    if mean:
        dist_to_S2 = dist_to_S2.mean()

    return dist_to_S2
