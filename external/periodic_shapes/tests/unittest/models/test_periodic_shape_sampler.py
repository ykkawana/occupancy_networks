import sys
sys.path.append('.')
from periodic_shapes.models import super_shape
from periodic_shapes.models import periodic_shape_sampler
from periodic_shapes.models import super_shape_sampler
from periodic_shapes import utils
from periodic_shapes.layers import super_shape_functions
import torch
from torch import nn
import math
from unittest import mock
import numpy as np

EPS = 1e-7


@mock.patch(
    'periodic_shapes.models.periodic_shape_sampler.PeriodicShapeSampler.get_periodic_net_r'
)
def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_2d(
    mocker):
    """Test if posed points are in correct place."""
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    dim = 2
    points_num = 5

    rotations = [[0.], [0.]]
    transitions = [[0., 0.], [1., 0.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        batch=batch)

    thetas_list = [math.pi / 2., 0., math.pi / 4.]
    P = len(thetas_list)
    thetas = torch.tensor(thetas_list).view(1, P, dim - 1).repeat(batch, 1, 1)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        thetas, preset_params)

    batched_points = torch.tensor([0.] * batch * points_num * dim).view(
        batch, points_num, dim)
    margin = 0.5
    mocker.return_value = torch.tensor([margin])
    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params, points=batched_points)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor(
        [[0., 1. + margin], [1. + margin, 0.],
         [0.5 + math.sqrt(margin**2 / 2),
          0.5 + math.sqrt(margin**2 / 2)]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    print(cartesian_coord[0])
    print(target_cartesian_coord)
    assert torch.allclose(cartesian_coord[0], target_cartesian_coord,
                          atol=EPS), (cartesian_coord[0],
                                      target_cartesian_coord)


@mock.patch(
    'periodic_shapes.models.periodic_shape_sampler.PeriodicShapeSampler.get_periodic_net_r'
)
def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_3d(
    mocker):
    """Test if posed points are in correct place."""
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    dim = 3
    points_num = 5

    rotations = [[0., 0., 0.], [0., 0., 0.]]
    transitions = [[0., 0., 0.], [1., 0., 0.]]
    linear_scales = [[1., 1., 1.]] * n

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim,
                                                          rational=True)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        batch=batch)

    thetas_list = [[math.pi / 2., 0.], [0., 0.], [math.pi / 4., 0.],
                   [math.pi, 0.], [math.pi / 2., math.pi / 2. - EPS]]
    P = len(thetas_list)
    thetas = torch.tensor(thetas_list).view(1, P, dim - 1).repeat(batch, 1, 1)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        thetas, preset_params)

    batched_points = torch.tensor([1.] * batch * points_num * dim).view(
        batch, points_num, dim)
    margin = 0.5
    mocker.return_value = torch.tensor([margin])
    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params, points=batched_points)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor(
        [[0., 1. + margin, 0.], [1. + margin, 0., 0.],
         [
             0.5 + math.sqrt(margin**2 / 2**2),
             0.5 + math.sqrt(margin**2 / 2**2), 0.
         ], [-1. - margin, 0., 0.], [0., 0., 1. + margin]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    print(cartesian_coord[0])
    print(target_cartesian_coord)
    assert torch.allclose(cartesian_coord[0],
                          target_cartesian_coord,
                          atol=EPS * 1e+1)


@mock.patch(
    'periodic_shapes.models.periodic_shape_sampler.PeriodicShapeSampler.get_periodic_net_r'
)
def test_transform_world_cartesian_coord_to_tsd_2d(mocker):
    """Test if area with positive signs is as expected size for all primitives 
    and each corner of shape (square) is as expected. 
    """
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    grid_range = [-2, 2]
    area_error_tol = 1e-1
    dim = 2
    points_num = 5

    rotations = [[0.], [math.pi / 2]]
    transitions = [[0., 0.], [0.5, 0.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        batch=batch)

    batched_theta_test_tensor = torch.tensor([theta] * batch * (dim - 1)).view(
        batch, 1, dim - 1)
    coord = utils.generate_grid_samples(grid_range,
                                        batch=batch,
                                        sample_num=sample_num,
                                        dim=dim)

    batched_points = torch.tensor([0.] * batch * points_num * dim).view(
        batch, points_num, dim)

    margin = torch.zeros([1]).float() + 0.2
    mocker.return_value = margin
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)

    sgn_bool = (sgn >= 0).float()
    assert [*sgn_bool.shape] == [batch, n, sample_num**dim]

    for idx in range(n):
        sgn_p = sgn_bool[0, idx, :]
        x = coord[0, :, 0][torch.where(sgn_p)]
        y = coord[0, :, 1][torch.where(sgn_p)]
        xmax = x.max()
        ymax = y.max()
        xmin = x.min()
        ymin = y.min()

        # Check corners
        assert torch.allclose(xmax,
                              torch.tensor([1.]) + transitions[idx][0] +
                              margin,
                              atol=area_error_tol)
        assert torch.allclose(xmin,
                              -torch.tensor([1.]) + transitions[idx][0] -
                              margin,
                              atol=area_error_tol)
        assert torch.allclose(ymax,
                              torch.tensor([1.]) + transitions[idx][1] +
                              margin,
                              atol=area_error_tol)
        assert torch.allclose(ymin,
                              -torch.tensor([1.]) + transitions[idx][1] -
                              margin,
                              atol=area_error_tol)


@mock.patch(
    'periodic_shapes.models.periodic_shape_sampler.PeriodicShapeSampler.get_periodic_net_r'
)
def test_transform_world_cartesian_coord_to_tsd_3d(mocker):
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5

    area_error_tol = 1e-1
    dim = 3
    device = 'cpu'

    margin = 0.15
    grid_ranges = [
        {
            'range': [[-1.2, 2.2], [-1.2, 1.2], [0, 0]],
            'sample_num': [sample_num, sample_num, 1]
        },
        {  # for y-z plane
            'range': [[0, 0], [-1.2, 1.2], [-1.2, 1.2]],
            'sample_num': [1, sample_num, sample_num]
        }
    ]
    target = [[{
        'xmin': -1.0 - margin,
        'xmax': 1.0 + margin,
        'ymin': -1.0 - margin,
        'ymax': 1.0 + margin,
        'zmin': 0.,
        'zmax': 0.
    }, {
        'xmin': 0.0 - margin,
        'xmax': 2.0 + margin,
        'ymin': -1.0 - margin,
        'ymax': 1.0 + margin,
        'zmin': 0.,
        'zmax': 0.
    }],
              [{
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -1.0 - margin,
                  'ymax': 1.0 + margin,
                  'zmin': -1.0 - margin,
                  'zmax': 1.0 + margin,
              }, {
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -1.0 - margin,
                  'ymax': 1.0 + margin,
                  'zmin': -1.0 - margin,
                  'zmax': 1.0 + margin
              }]]
    rotationss = [[[0., 0., 0.], [0., math.pi / 2, 0.]]] * len(grid_ranges)
    translationss = [[[0., 0., 0.], [1.0, 0., 0.]], [[0., 0., 0.],
                                                     [0., 0., 0.]]]
    linear_scaless = [[[1., 1., 1.], [1., 1., 1.]]] * len(grid_ranges)
    for idx in range(len(grid_ranges)):
        rotations = rotationss[idx]
        transitions = translationss[idx]
        linear_scales = linear_scaless[idx]

        sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                              m,
                                                              n,
                                                              dim=dim)
        sampler.to(device)
        preset_params = utils.generate_multiple_primitive_params(
            m,
            n1,
            n2,
            n3,
            a,
            b,
            rotations_angle=rotations,
            transitions=transitions,
            linear_scales=linear_scales,
            batch=batch)

        for param in preset_params:
            if not preset_params[param] is None:
                preset_params[param] = preset_params[param].to(device)

        grid_range = grid_ranges[idx]
        coord = utils.generate_grid_samples(grid_range,
                                            batch=batch,
                                            dim=dim,
                                            device=device)

        batched_points = torch.zeros([batch, points_num, dim]).float()
        mocker.return_value = torch.zeros([1]).float() + margin
        sgn = sampler.transform_world_cartesian_coord_to_tsd(
            coord, preset_params, points=batched_points)

        sgn_bool = (sgn >= 0).float()
        assert [*sgn_bool.shape
                ] == [batch, n, np.prod(grid_range['sample_num'])]

        for idx2 in range(n):
            sgn_p = sgn_bool[0, idx2, :]
            x = coord[0, :, 0][torch.where(sgn_p)]
            y = coord[0, :, 1][torch.where(sgn_p)]
            z = coord[0, :, 2][torch.where(sgn_p)]
            xmax = x.max()
            ymax = y.max()
            zmax = z.max()
            xmin = x.min()
            ymin = y.min()
            zmin = z.min()

            target_coord = target[idx][idx2]
            target_coord = {
                key: torch.tensor(target_coord[key])
                for key in target_coord
            }

            assert torch.allclose(xmax,
                                  target_coord['xmax'],
                                  atol=area_error_tol)
            assert torch.allclose(xmin,
                                  target_coord['xmin'],
                                  atol=area_error_tol)
            assert torch.allclose(ymax,
                                  target_coord['ymax'],
                                  atol=area_error_tol)
            assert torch.allclose(ymin,
                                  target_coord['ymin'],
                                  atol=area_error_tol)
            assert torch.allclose(zmax,
                                  target_coord['zmax'],
                                  atol=area_error_tol)
            assert torch.allclose(zmin,
                                  target_coord['zmin'],
                                  atol=area_error_tol)


def test_forward_3d():
    batch = 2
    m = 4
    n = 5
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    dim = 3
    points_num = 2

    rotations = [[0., 0., 0.]] * n
    transitions = [[0., 0., 0.]] * n
    linear_scales = [[1., 1., 1.]] * n

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        batch=batch)

    grid_range = {
        'range': [[-math.pi, math.pi], [-math.pi / 2, math.pi / 2]],
        'sample_num': [5, 5]
    }
    batched_theta_test_tensor = utils.generate_grid_samples(grid_range,
                                                            batch=batch,
                                                            dim=dim - 1)

    coord_grid_range = 5
    batched_coord_test_tensor = utils.generate_grid_samples(coord_grid_range,
                                                            batch=batch,
                                                            dim=dim)

    batched_points = torch.zeros([batch, points_num, dim]).float()
    sampler.forward(preset_params,
                    points=batched_points,
                    coord=batched_coord_test_tensor,
                    thetas=batched_theta_test_tensor)


def test_forward_2d():
    batch = 2
    m = 4
    n = 5
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    dim = 2
    points_num = 2

    rotations = [[0.]] * n
    transitions = [[0., 0.]] * n
    linear_scales = [[1., 1.]] * n

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        batch=batch)

    grid_range = {
        'range': [[-math.pi, math.pi], [-math.pi / 2, math.pi / 2]],
        'sample_num': [5, 5]
    }
    batched_theta_test_tensor = utils.generate_grid_samples(
        grid_range, batch=batch, dim=dim)[:, :, 0].unsqueeze(-1)

    coord_grid_range = 5
    batched_coord_test_tensor = utils.generate_grid_samples(coord_grid_range,
                                                            batch=batch,
                                                            dim=dim)

    batched_points = torch.zeros([batch, points_num, dim]).float()
    sampler.forward(preset_params,
                    points=batched_points,
                    coord=batched_coord_test_tensor,
                    thetas=batched_theta_test_tensor)


def test_transform_world_cartesian_coord_to_tsd_2d_with_periodic_net():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5

    dim = 2

    rotations = [[0.]]
    transitions = [[0., 0.]]
    linear_scales = [[1., 1.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          last_scale=.1,
                                                          dim=dim)

    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=sample_num, batch=batch, dim=dim)
    batched_points = torch.zeros([batch, points_num, dim]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)

    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.min(), sgn.max())


def test_transform_world_cartesian_coord_to_tsd_3d_with_periodic_net():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 10

    dim = 3

    #rotations = [[0., 0., 0.]]
    rotations = [[0., math.pi / 2., math.pi / 2.]]
    transitions = [[0., 0., 0.]]
    linear_scales = [[1., 1., 1.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          last_scale=.1,
                                                          dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=P, batch=batch, sgn_convertible=True, dim=dim)

    batched_points = torch.ones([batch, points_num, dim]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)
    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.min(), sgn.max())


def test_test_decoder_consistency_ShapeDecoderCBatchNorm():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 10

    dim = 3

    #rotations = [[0., 0., 0.]]
    rotations = [[0., math.pi / 2., math.pi / 2.]]
    transitions = [[10., 10., 0.]]
    linear_scales = [[1., 1., 1.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(
        points_num,
        m,
        n,
        decoder_class='ShapeDecoderCBatchNorm',
        last_scale=.1,
        no_encoder=True,
        is_feature_coord=False,
        dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=P, batch=batch, sgn_convertible=True, dim=dim)

    batched_points = torch.rand([batch, points_num]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)
    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.min(), sgn.max())


def test_decoder_consistency_PrimitiveWiseGroupConvDecoder():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 30

    dim = 3

    #rotations = [[0., 0., 0.]]
    rotations = [[0., math.pi / 2., math.pi / 2.]]
    #rotations = [[0., math.pi / 2., math.pi / 2.]]
    transitions = [[0.0, 10.0, 0.]]
    linear_scales = [[1., 1., 1.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(
        points_num,
        m,
        n,
        decoder_class='PrimitiveWiseGroupConvDecoder',
        #decoder_class='BatchNormDecoder',
        last_scale=.1,
        no_encoder=True,
        is_feature_angles=False,
        is_feature_coord=True,
        is_feature_radius=False,
        dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=P, batch=batch, sgn_convertible=True, dim=dim)

    batched_points = torch.ones([batch, points_num]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)
    print((sgn.abs() > 1e-5).sum(), (sgn[sgn.abs() > 1e-5]).abs().mean(),
          sgn.shape)
    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.max(), sgn.min(), sgn.median(),
                                       sgn.mean())


def test_decoder_consistency_OtherDecoders():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 10

    dim = 3

    rotations = [[0., 0., 0.]] * n
    #rotations = [[0., math.pi / 2, 0.]] * n
    #rotations = [[0., math.pi / 2., math.pi / 2.],
    #             [0., math.pi / 2., math.pi / 4.], [0., math.pi, math.pi / 2.]]
    transitions = [[0., 0., 0.]] * n
    #transitions = [[0., 1., 0.], [0., 1., 10.], [0., 1., 1.]]
    linear_scales = [[1., 1., 1.]] * n
    #linear_scales = [[1., 1., 1.], [0.2, 0.8, 1.2], [1., 1.2, 2.2]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(
        points_num,
        m,
        n,
        #decoder_class='MLPDecoder',
        #decoder_class='BatchNormDecoder',
        decoder_class='PrimitiveWiseGroupConvDecoder',
        #decoder_class='BatchNormDecoderSharedWeight',
        #decoder_class='PrimitiveWiseGroupConvDecoderLegacy',
        last_scale=10,
        no_encoder=True,
        is_shape_sampler_sphere=True,
        spherical_angles=True,
        is_feature_coord=True,
        is_feature_angles=False,
        is_feature_radius=False,
        dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=P, batch=batch, sgn_convertible=True, dim=dim)

    batched_points = torch.rand([batch, points_num]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coords = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points)

    print('coord mean main', coords.mean(), (coords**2).sum(-1).sqrt().mean())

    print('check coord')
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coords.view(
        batch, -1, dim),
                                                         preset_params,
                                                         points=batched_points)
    print('check coord done')
    for idx in range(n):
        coord = coords[:, idx, :, :]
        sgn = sampler.transform_world_cartesian_coord_to_tsd(
            coord, preset_params, points=batched_points)[:, idx, :]
        print(sgn.shape)
        assert torch.allclose(sgn, torch.zeros_like(sgn),
                              atol=1e-5), (sgn.max(), sgn.min(), sgn.median(),
                                           sgn.mean())


def test_decoder_sdf_consistency_OtherDecoders():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 10

    dim = 3

    rotations = [[0., 0., 0.]] * n
    #rotations = [[0., math.pi / 2, 0.]] * n
    #rotations = [[0., math.pi / 2., math.pi / 2.],
    #             [0., math.pi / 2., math.pi / 4.], [0., math.pi, math.pi / 2.]]
    transitions = [[0., 0., 0.]] * n
    #transitions = [[0., 1., 0.], [0., 1., 10.], [0., 1., 1.]]
    linear_scales = [[1., 1., 1.]] * n
    #linear_scales = [[1., 1., 1.], [0.2, 0.8, 1.2], [1., 1.2, 2.2]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(
        points_num,
        m,
        n,
        #decoder_class='MLPDecoder',
        #decoder_class='BatchNormDecoder',
        decoder_class='PrimitiveWiseGroupConvDecoder',
        #decoder_class='BatchNormDecoderSharedWeight',
        #decoder_class='PrimitiveWiseGroupConvDecoderLegacy',
        last_scale=10,
        no_encoder=True,
        is_shape_sampler_sphere=True,
        spherical_angles=False,
        is_feature_coord=True,
        is_feature_angles=False,
        is_feature_radius=False,
        return_sdf=False,
        dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n,
        batch=batch)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=P, batch=batch, sgn_convertible=True, dim=dim)

    batched_points = torch.rand([batch, points_num]).float()

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)
    # B, P, dim
    coords = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor,
        radius,
        preset_params,
        points=batched_points)

    print('coord mean main', coords.mean(), (coords**2).sum(-1).sqrt().mean())

    print('check coord')
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coords.view(
        batch, -1, dim),
                                                         preset_params,
                                                         points=batched_points)
    print('check coord done')
    for idx in range(n):
        coord = coords[:, idx, :, :]
        sgn = sampler.transform_world_cartesian_coord_to_tsd(
            coord, preset_params, points=batched_points)[:, idx, :]
        print(sgn.shape)
        assert torch.allclose(sgn, torch.zeros_like(sgn),
                              atol=1e-5), (sgn.max(), sgn.min(), sgn.median(),
                                           sgn.mean())
