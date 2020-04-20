import numpy as np
import torch
from pykeops.torch import LazyTensor
from torch.nn import functional as F


def create_planar_mesh(length):

    assert length >= 2
    width = height = length
    height = int(height)
    width = int(width)
    X, Y = np.meshgrid(range(width), range(height))
    h = np.stack([X.reshape(-1), Y.reshape(-1)])
    vertices = h.T
    vertices = vertices / (length - 1)
    assert vertices.max() == 1 and vertices.min() == 0

    faces = []

    for y in range(height - 1):
        for x in range(width - 1):
            faces.append(
                [x + width * y, x + 1 + width * y, x + 1 + width * (y + 1)])
            faces.append(
                [x + width * y, x + 1 + width * (y + 1), x + width * (y + 1)])
    faces = np.asarray(faces)

    return vertices, faces


def chamfer_loss(source_points, target_points, mode='l2'):
    """
    N = n_primitives
    B = batch
    Ps = points per primitive
    Pt = target points

    Args:
        primitive_points: B x N x Ps x 2
        target_points: B x Pt x 2
        prob: B x N
        surface_mask: B x N x Ps
        p (str): original = (x.abs ** 2).sum() ** 0.5 or euclidean
    """

    B, P1, D1 = source_points.shape
    _, P2, D2 = target_points.shape

    G_i1 = LazyTensor(source_points.unsqueeze(2))
    X_j1 = LazyTensor(target_points.unsqueeze(1))

    if mode == 'l2':
        dist = (G_i1 - X_j1).sqnorm2()
    elif mode == 'l1':
        dist = (G_i1 - X_j1).norm2()
    else:
        raise NotImplementedError

    # B x (Ps * N)
    idx1 = dist.argmin(dim=2)
    target_points_selected = batched_index_select(target_points, 1,
                                                  idx1.view(B, -1))
    diff_primitives2target = source_points - target_points_selected

    if mode == 'l2':
        loss_source2target = (diff_primitives2target**2).sum(-1)
    elif mode == 'l1':
        loss_source2target = torch.norm(diff_primitives2target, None,
                                        dim=2).squeeze(-1)

    idx2 = dist.argmin(dim=1)  # Grid

    source_points_selected = batched_index_select(source_points, 1,
                                                  idx2.view(B, -1))
    diff_target2source = source_points_selected - target_points
    if mode == 'l2':
        loss_target2source = (diff_target2source**2).sum(-1)
    elif mode == 'l1':
        loss_target2source = torch.norm(diff_target2source, None,
                                        dim=2).squeeze(-1)
    return loss_target2source.mean() + loss_source2target.mean()


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
