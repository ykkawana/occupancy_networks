import torch
from torch import nn
from periodic_shapes.external import gumbel_softmax

EPS = 1e-7


def apply_rotation(coord, rotation, inv=False):
    B, N, P, dim = coord.shape
    _, N2, dim2 = rotation.shape
    assert N == N2, (N, N2)
    if dim == 2:
        assert dim2 == 1
        return apply_2d_rotation(coord, rotation, inv=inv)
    elif dim == 3:
        assert dim2 == 4
        return apply_3d_rotation(coord, rotation, inv=inv)
    else:
        raise NotImplementedError


def apply_2d_rotation(xy, rotation, inv=False):
    B, N, P, dim = xy.shape
    # B, n_primitives, P, 2, 2
    rotation_matrix = get_2d_rotation_matrix(rotation, inv=inv).view(
        B, N, 1, 2, 2).repeat(1, 1, P, 1, 1)
    assert not torch.isnan(rotation_matrix).any()
    # B, n_primitives, P, 2, 1
    xy_before_rotation = xy.view(B, N, P, 2, 1)
    rotated_xy = torch.bmm(rotation_matrix.view(-1, 2, 2),
                           xy_before_rotation.view(-1, 2, 1)).view(B, N, P, 2)
    return rotated_xy


def apply_3d_rotation(coord, rotation, inv=False):
    B, N, P, dim = coord.shape
    B2, N2, D = rotation.shape
    assert B == B2
    assert N == N2, (N, N2)
    # rotation quaternion in [w, x, y, z]

    q = rotation
    if inv:
        w, x, y, z = q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    qq = rotation**2
    w2, x2, y2, z2 = qq[..., 0], qq[..., 1], qq[..., 2], qq[..., 3]
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                          dim=-1).view(B2, N2, 1, 3, 3)

    rotated_coord = (coord.unsqueeze(-2) * rot_mat).sum(-1)
    return rotated_coord


def get_probabilistic_m_vector(logits, dim=2, hard=False):
    """Return weight vector for m coefficient probabilistically.
    Args:
        logits (1, n_primitives, max_m, self.dim-1)
        dim: which column is for max_m coeffient
        hard (bool): return one hot vector if true.
    Return:
        selector  (1, n_primitives, max_m, self.dim-1)
    """
    assert len(logits.shape) == 4
    selector = gumbel_softmax(nn.functional.relu(logits + EPS),
                              dim=dim,
                              hard=hard)
    return selector


def get_deterministic_m_vector(logits, dim=2):
    """Return weight vector for m coefficient probabilistically.
    Args:
        logits (1, n_primitives, max_m, self.dim-1)
        dim: which column is for max_m coeffient
    Return:
        selector  (1, n_primitives, max_m, self.dim-1)
    """
    assert len(logits.shape) == 4, logits.shape
    y = nn.functional.relu(logits + EPS).argmax(axis=dim, keepdim=True)
    #y_onehot = torch.FloatTensor(1, self.n_primitives, self.max_m, 1)
    #y_onehot = y_onehot.zero_().to(y.get_device())
    y_onehot = torch.zeros_like(logits)

    y_onehot.scatter_(dim, y, 1)
    return y_onehot


def get_m_vector(logits, probabilistic=True):
    if probabilistic:
        m_vector = get_probabilistic_m_vector(logits)
    else:
        m_vector = get_deterministic_m_vector(logits)
    return m_vector


def get_2d_rotation_matrix(rotation, inv=False):
    sgn = -1. if inv else 1.
    upper_rotation_matrix = torch.cat([rotation.cos(), -sgn * rotation.sin()],
                                      axis=-1)
    lower_rotation_matrix = torch.cat(
        [sgn * rotation.sin(), rotation.cos()], axis=-1)
    rotation_matrix = torch.stack(
        [upper_rotation_matrix, lower_rotation_matrix], axis=-2)
    return rotation_matrix


def convert_tsd_range_to_zero_to_one(tsd, scale=100):
    return nn.functional.sigmoid(tsd * scale)
