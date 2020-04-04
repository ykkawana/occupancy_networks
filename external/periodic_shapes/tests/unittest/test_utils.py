import utils
import torch
import numpy as np


def test_sphere_cartesian2polar():
    coord = torch.tensor([0., 0., 0.]).view(1, 3)
    r, angles = utils.sphere_cartesian2polar(coord)

    # Test if proper shape for 3d
    assert len(r.shape) == 2
    assert len(angles.shape) == 2
    assert angles.shape[-1] == 2

    # Test if nan occurs
    assert not torch.any(torch.isnan(r))
    assert not torch.any(torch.isnan(angles))

    coord = torch.tensor([
        0.,
        0.,
    ]).view(1, 2)
    r, angles = utils.sphere_cartesian2polar(coord)

    # Test if proper shape for 2d
    assert len(angles.shape) == 2
    assert angles.shape[-1] == 1

    coord = torch.tensor([1., 0., 0.])
    r, angles = utils.sphere_cartesian2polar(coord)

    # Check proper value
    assert r.item() == 1.
    assert angles[0].item() == 0.
    assert angles[1].item() == 0.

def test_generate_grid_samples():
    # grid
    sampling = 'grid'
    batch = 2
    sample_num = 3
    grid_size = 5
    dim = 1 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]
    assert torch.all(torch.eq(coord[0], coord[1]))

    batch = 2
    sample_num = 3
    grid_size = 5
    dim = 2 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]
    assert torch.all(torch.eq(coord[0], coord[1]))
    assert torch.all((-grid_size <= coord) & (coord <= grid_size))

    batch = 2
    sample_num = 3
    grid_size = [-1, 1]
    dim = 2 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]
    assert torch.all(torch.eq(coord[0], coord[1]))
    assert torch.all((grid_size[0] <= coord) & (coord <= grid_size[1]))

    batch = 2
    grid_size = {
        'range': [[2, 4], [-1, 1]],
        'sample_num': [3, 5]
    }
    dim = len(grid_size['range'])
    total_sample_num = np.prod(grid_size['sample_num'])
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, total_sample_num, dim]
    assert torch.all(torch.eq(coord[0], coord[1]))

    for idx in range(dim):
        assert torch.all((grid_size['range'][idx][0] <= coord[:, :, idx]) & (coord[:, :, idx] <= grid_size['range'][idx][1]))

    # uniform 
    sampling = 'uniform'
    batch = 2
    sample_num = 3
    grid_size = 5
    dim = 1 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]

    batch = 2
    sample_num = 3
    grid_size = 5
    dim = 2 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]
    assert torch.all((-grid_size <= coord) & (coord <= grid_size))

    batch = 2
    sample_num = 3
    grid_size = [-1, 1]
    dim = 2 
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, sample_num ** dim, dim]
    assert torch.all((grid_size[0] <= coord) & (coord <= grid_size[1]))

    batch = 2
    grid_size = {
        'range': [[2, 4], [-1, 1]],
        'sample_num': [3, 5]
    }
    dim = len(grid_size['range'])
    total_sample_num = np.prod(grid_size['sample_num'])
    coord = utils.generate_grid_samples(grid_size, batch=batch, sampling=sampling, sample_num=sample_num, dim=dim)
    assert [*coord.shape] == [batch, total_sample_num, dim]

    for idx in range(dim):
        assert torch.all((grid_size['range'][idx][0] <= coord[:, :, idx]) & (coord[:, :, idx] <= grid_size['range'][idx][1]))


