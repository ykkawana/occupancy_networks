import torch
import math
from collections import defaultdict
from periodic_shapes.external.QuaterNet.common import quaternion
import numpy as np
import warnings

EPS = 1e-7


def safe_atan(y, x):
    theta = torch.atan2(y, x)
    #theta = torch.atan(y / (x + EPS * 1e-3))
    #theta = torch.where(theta >= 0, theta, (2 * math.pi + theta))
    return theta


def sphere_polar2cartesian(radius, angles):
    assert len(radius.shape) == len(angles.shape)
    assert radius.shape[-1] == 1
    theta = angles[..., 0]
    phi = torch.zeros(
        [1], device=theta.device) if angles.shape[-1] == 1 else angles[..., 1]

    x = radius.squeeze(-1) * theta.cos() * phi.cos()
    y = radius.squeeze(-1) * theta.sin() * phi.cos()
    coord = [x, y]
    if angles.shape[-1] == 2:
        z = phi.sin() * radius.squeeze(-1)
        coord.append(z)
    return torch.stack(coord, axis=-1)


def sphere_cartesian2polar(coord):
    r = ((coord**2).sum(-1) + EPS).sqrt().unsqueeze(-1)
    theta = torch.atan2(coord[..., 1], coord[..., 0]).unsqueeze(-1)
    angles = [theta]
    if coord.shape[-1] == 3:
        phi = torch.atan(coord[..., 2].unsqueeze(-1) * r * theta.sin() /
                         (coord[..., 1].unsqueeze(-1) + EPS))
        angles.append(phi)
    angles = torch.cat(angles, axis=-1)
    return r, angles


def supershape_angles(radius, coord, angles):
    theta = safe_atan(coord[..., 1], coord[..., 0]).unsqueeze(-1)
    new_angles = [theta]

    if coord.shape[-1] == 3:
        phi = safe_atan(coord[..., 2] * radius[..., 0] * angles[..., 0].sin(),
                        coord[..., 1]).unsqueeze(-1)
        new_angles.append(phi)

    return torch.cat(new_angles, axis=-1)


def generate_grid_samples(grid_size,
                          batch=1,
                          sampling='grid',
                          sample_num=100,
                          device='cpu',
                          dim=2):
    """Generate mesh grid.
    Args:
        grid_size (int, list, dict):
            if type is int, sample between [-grid_size, grid_size] range for all
            dims.
            if type is list, sample between grid_size for all dims. 
            if type is dict, For each dim, sample between each element of `range` in grid_size 
            with number of samples specified by `sample_num` in grid_size.
        batch (int): batch size. Sample size will be batch size * grid samples
        sampling (str): sample gridwise if it's `grid`, sample by uniform dist. if it's `uniform`.
        sample_num (int): number of samples. If grid_type is dict, this argument will be ignored.
        device: if set, data will be initialized on that device. otherwise will be stored on `cpu`.
        dim: dimension of meshgrid. Defaults to 2.
    Returns:
        coord (B, sample size, dim)
    """
    if isinstance(grid_size, list):
        sampling_start, sampling_end = grid_size
    elif isinstance(grid_size, int):
        sampling_start, sampling_end = -grid_size, grid_size
    else:
        dim = len(grid_size['range'])
        assert dim == len(grid_size['sample_num'])
        warnings.warn('sample_num and dim will be ignored')

    def contiguous_batch(s):
        return s.contiguous().view(1, -1).repeat(batch, 1)

    ranges = []

    if sampling == 'grid':
        for idx in range(dim):
            if isinstance(grid_size, dict):
                sampling_start, sampling_end = grid_size['range'][idx]
                sample_num = grid_size['sample_num'][idx]
            sampling_points = torch.linspace(sampling_start,
                                             sampling_end,
                                             sample_num,
                                             device=device)
            ranges.append(sampling_points)

        coord_list = [contiguous_batch(s) for s in torch.meshgrid(ranges)]
        return torch.stack(coord_list, axis=-1)
    elif sampling == 'uniform':
        if isinstance(grid_size, dict):
            total_sample_num = np.prod(grid_size['sample_num'])
        else:
            total_sample_num = sample_num**dim
        sampling_points = torch.empty([batch, total_sample_num, dim],
                                      device=device).uniform_(0, 1)

        def normalize(x, bounds):
            return bounds[0] + (x - 0.) * (bounds[1] - bounds[0]) / (1. - 0.)

        if isinstance(grid_size, int):
            return sampling_points * grid_size
        elif isinstance(grid_size, list):
            return normalize(sampling_points, grid_size)
        elif isinstance(grid_size, dict):
            ranged_sampling_points_list = []
            for idx in range(dim):
                ranged_sampling_points_list.append(
                    normalize(sampling_points[..., idx],
                              grid_size['range'][idx]))
            return torch.stack(ranged_sampling_points_list, axis=-1)
    else:
        raise NotImplementedError('no such sampling mode')


def sample_spherical_angles(batch=1,
                            sampling='grid',
                            sample_num=100,
                            device='cpu',
                            phi_margin=EPS,
                            theta_margin=0,
                            sgn_convertible=False,
                            spherical_coord=False,
                            dim=2):
    assert dim in [2, 3]
    if not sgn_convertible:
        phi_margin = 0
    grid_range = {'range': [[-math.pi, math.pi]], 'sample_num': [sample_num]}
    if dim == 2:
        grid_range = {
            'range': [[-math.pi, math.pi]],
            'sample_num': [sample_num]
        }
        #grid_range = {'range': [[0., 2 * math.pi]], 'sample_num': [sample_num]}
    else:
        grid_range = {
            'range': [[-math.pi + theta_margin, math.pi - theta_margin],
                      [-math.pi / 2 + phi_margin, math.pi / 2 - phi_margin]],
            'sample_num': [sample_num, sample_num]
        }
    if spherical_coord:
        assert dim == 3
        grid_range = {
            'range': [[0, 2 * math.pi], [0, math.pi]],
            'sample_num': [sample_num, sample_num]
        }
    angles = generate_grid_samples(grid_range,
                                   sampling=sampling,
                                   device=device,
                                   batch=batch,
                                   dim=dim - 1)
    return angles


def get_single_input_element(theta, m, n1, n2, n3, a, b):
    n1inv = 1. / n1
    ainv = 1. / a
    binv = 1. / b
    scatter_point_size = 5
    theta_test_tensor = torch.tensor([theta])
    m_tensor = torch.tensor([m])
    n1inv_tensor = torch.tensor([n1inv])
    n2_tensor = torch.tensor([n2])
    n3_tensor = torch.tensor([n3])
    ainv_tensor = torch.tensor([ainv])
    binv_tensor = torch.tensor([binv])
    return (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
            ainv_tensor, binv_tensor)


def generate_single_primitive_params(m,
                                     n1,
                                     n2,
                                     n3,
                                     a,
                                     b,
                                     rotation=0.,
                                     transition=[
                                         0.,
                                         0.,
                                     ],
                                     linear_scale=[1., 1.],
                                     logit=None,
                                     dim=2,
                                     batch=1):
    assert dim in [2, 3]
    n1 = torch.tensor([1. / n1]).float().view(1, 1,
                                              1).repeat(batch, 1, dim - 1)
    n2 = torch.tensor([n2]).float().view(1, 1, 1).repeat(batch, 1, dim - 1)
    n3 = torch.tensor([n3]).float().view(1, 1, 1).repeat(batch, 1, dim - 1)
    a = torch.tensor([1. / a]).float().view(1, 1, 1).repeat(batch, 1, dim - 1)
    b = torch.tensor([1. / b]).float().view(1, 1, 1).repeat(batch, 1, dim - 1)
    transition = torch.tensor(transition).float().view(1, 1, dim).repeat(
        batch, 1, 1)
    if dim == 2:
        rotation = torch.tensor(rotation).float().view(1, 1,
                                                       1).repeat(batch, 1, 1)
    else:
        rotation = torch.tensor(rotation).float().view(1, 1,
                                                       4).repeat(batch, 1, 1)
    linear_scale = torch.tensor(linear_scale).float().view(1, 1, dim).repeat(
        batch, 1, 1)

    if logit:
        m_vector = linear_scale = torch.tensor(logit).float().view(
            1, 1, -1, 1).repeat(batch, 1, 1, dim - 1)
    else:
        logit = [0.] * (m + 1)
        logit[m] = 1.
        m_vector = torch.tensor(logit).float().view(1, 1, m + 1, 1).repeat(
            batch, 1, 1, dim - 1)

    return {
        'n1': n1,
        'n2': n2,
        'n3': n3,
        'a': a,
        'b': b,
        'm_vector': m_vector,
        'rotation': rotation,
        'transition': transition,
        'linear_scale': linear_scale,
        'prob': None
    }


def generate_multiple_primitive_params(m,
                                       n1,
                                       n2,
                                       n3,
                                       a,
                                       b,
                                       rotations_angle=[
                                           [0.],
                                           [0.],
                                       ],
                                       transitions=[[
                                           0.,
                                           0.,
                                       ], [
                                           0.,
                                           0.,
                                       ]],
                                       linear_scales=[[1., 1.], [1., 1.]],
                                       nn=None,
                                       logit=None,
                                       batch=1):
    params = defaultdict(lambda: [])
    n = len(transitions)
    assert len(transitions) == len(rotations_angle), len(linear_scales)
    dim = len(transitions[0])
    assert dim in [2, 3]

    if dim == 2:
        assert len(rotations_angle[0]) == 1
        rotations = rotations_angle
    else:
        assert len(rotations_angle[0]) == 3
        rotations = convert_angles_to_quaternions(rotations_angle)

    for idx in range(n):
        if not nn is None and idx > nn - 1:
            break
        param = generate_single_primitive_params(
            m,
            n1,
            n2,
            n3,
            a,
            b,
            rotation=rotations[idx],
            transition=transitions[idx],
            linear_scale=linear_scales[idx],
            logit=None,
            dim=dim,
            batch=batch)
        for key in param:
            params[key].append(param[key])
    return_param = {}
    for key in params:
        if key == 'prob':
            continue
        return_param[key] = torch.cat(params[key], axis=1)
    return_param['prob'] = None

    return return_param


def convert_angles_to_quaternions(rotations_angle):
    rotations = []
    for rotation in rotations_angle:
        rotations.append(
            quaternion.euler_to_quaternion(np.array(rotation), 'xyz').tolist())
    return rotations
