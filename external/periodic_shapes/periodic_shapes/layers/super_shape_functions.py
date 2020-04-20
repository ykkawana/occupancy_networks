import torch

EPS = 1e-7


def polar2cartesian(radius, angles):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        r: radius (B, N, P, 1 or 2) last dim is one in 2D mode, two in 3D.
        angles: angle (B, 1, P, 1 or 2) 
    """
    #print('radius angles in s2c mean', radius[..., -1].mean(), angles.mean())
    dim = radius.shape[-1]
    dim2 = angles.shape[-1]
    P = radius.shape[-2]
    P2 = angles.shape[-2]
    assert dim == dim2
    assert dim in [1, 2]
    assert P == P2

    theta = angles[..., 0]
    phi = torch.zeros([1], device=angles.device) if dim == 1 else angles[...,
                                                                         1]
    r1 = radius[..., 0]
    r2 = torch.ones([1], device=radius.device) if dim == 1 else radius[..., 1]

    phicosr2 = phi.cos() * r2
    cartesian_coord_list = [
        theta.cos() * r1 * phicosr2,
        theta.sin() * r1 * phicosr2
    ]

    # 3D
    if dim == 2:
        cartesian_coord_list.append(phi.sin() * r2)
    return torch.stack(cartesian_coord_list, axis=-1)


def sphere2cartesian(radius, angles):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        r: radius (B, N, P, 1 or 2) last dim is one in 2D mode, two in 3D.
        angles: angle (B, 1, P, 1 or 2) 
    """
    #print('radius angles in s2c mean', radius[..., 0].mean(), angles.mean())
    dim = radius.shape[-1]
    dim2 = angles.shape[-1]
    P = radius.shape[-2]
    P2 = angles.shape[-2]
    assert dim2 in [1, 2]
    assert P == P2

    theta = angles[..., 0]
    phi = torch.zeros([1], device=angles.device) if dim == 1 else angles[...,
                                                                         1]
    r = radius[..., 0]

    phicosr2 = phi.cos() * r
    cartesian_coord_list = [
        theta.cos() * r * phicosr2,
        theta.sin() * r * phicosr2
    ]

    # 3D
    if dim == 2:
        cartesian_coord_list.append(phi.sin() * r)
    return torch.stack(cartesian_coord_list, axis=-1)


def cartesian2sphere(coord):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        coord: (B, N, P, D)
    """
    dim = coord.shape[-1]
    x = coord[..., 0]
    y = coord[..., 1]
    z = torch.zeros([1], device=coord.device) if dim == 2 else coord[..., 2]
    x_non_zero = torch.where(x == 0, x + EPS, x)
    theta = torch.atan2(y, x_non_zero)

    assert not torch.isnan(theta).any(), (theta)
    r = (coord**2).sum(-1).clamp(min=EPS).sqrt()
    assert not torch.isnan(r).any(), (r)

    xysq_non_zero = (x**2 + y**2).clamp(min=EPS).sqrt()
    #xysq_non_zero = torch.where(xysq == 0, EPS + xysq, xysq)
    #xysq_non_zero = xysq.clamp(min=EPS)
    #phi = torch.atan(z / xysq_non_zero)
    phi = torch.atan2(z, xysq_non_zero)

    assert not torch.isnan(phi).any(), (phi)

    # (B, N, P)
    return r.unsqueeze(-1).expand([*coord.shape[:-1],
                                   dim - 1]), torch.stack([theta, phi],
                                                          axis=-1)


def rational_supershape(theta, sin, cos, n1, n2, n3, a, b):
    def U(theta):
        u = (a * cos).abs()
        assert not torch.isnan(u).any(), (u, a)
        return u

    def V(theta):
        v = (b * sin).abs()
        assert not torch.isnan(v).any()
        return v

    def W(theta):
        w = (U(theta) /
             (n2 +
              (1. - n2) * U(theta) + EPS)) + (V(theta) /
                                              (n3 +
                                               (1. - n3) * V(theta) + EPS))
        assert not torch.isnan(w).any(), (n2, n3)
        return w

    #r = (2.**(-(n1 + EPS))) * (n1 / (W(theta) + EPS) + 1. - n1)
    r = 1. / (n1 * W(theta) + 1. - n1 + EPS)
    assert not torch.isnan(r).any(), n1
    return r


def supershape(theta, sin, cos, n1, n2, n3, a, b):
    def U(theta):
        u = (a * cos).abs()**n2
        assert not torch.isnan(u).any(), (n2, a)
        return u

    def V(theta):
        v = (b * sin).abs()**n3
        assert not torch.isnan(v).any(), (n3, b)
        return v

    r = (U(theta) + V(theta) + EPS)**(-n1)
    assert not torch.isnan(r).any(), (r, n1)
    return r
