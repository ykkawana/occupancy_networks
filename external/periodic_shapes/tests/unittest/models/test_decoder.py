import sys
sys.path.append('.')
import torch
from periodic_shapes.models import decoder, periodic_shape_sampler


def test_encoder_decoder():
    batch = 3
    m = 3
    n = 1
    dim = 3
    points_num = 7
    angles_num = 5

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          last_scale=1.,
                                                          dim=dim)

    x = torch.zeros([batch, points_num, dim]).float()
    for N in [1, n]:
        # B, feature_dim
        encoded = sampler.encoder(x).view(batch, 1, 1,
                                          -1).repeat(1, angles_num, n, 1)
        thetas = torch.zeros([batch, N, angles_num, dim - 1]).float()
        radius = torch.zeros([batch, n, angles_num]).float()
        decoded = sampler.decoder(encoded, thetas, radius)

        assert [*decoded.shape] == [batch, n, angles_num, 1]
