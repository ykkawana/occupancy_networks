import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from periodic_shapes.models import model_utils
import plotly.graph_objects as go


def tensor2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def check_and_reduce_batch(points, last_dim, ndim_range):
    assert points.ndim in ndim_range
    if not last_dim is None:
        assert points.shape[-1] == last_dim, (points.shape, last_dim)
    if points.ndim == max(ndim_range):
        warnings.warn('only draw sample in first batch')
        points = points[0, ...]
    return points


def plot_primitive_point_cloud_2d(plot, points, s=1, title=''):
    points_numpy = tensor2numpy(points)

    points_numpy = check_and_reduce_batch(points_numpy, 2, [3, 4])

    n = points_numpy.shape[0]
    for idx in range(n):
        x = points_numpy[idx, :, 0]
        y = points_numpy[idx, :, 1]
        if title:
            plot.set_title(title)
        plot.set_aspect('equal')
        plot.scatter(x, y, s=s)


def plot_primitive_point_cloud_3d(points, s=1, title=''):
    points_numpy = tensor2numpy(points)
    points_numpy = check_and_reduce_batch(points_numpy, 3, [3, 4])

    plots = []
    marker_opt = dict(size=s)
    n = points_numpy.shape[0]
    for idx in range(n):
        x = points_numpy[idx, :, 0]
        y = points_numpy[idx, :, 1]
        z = points_numpy[idx, :, 2]
        plots.append(
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker_opt))
    fig = go.Figure(data=plots)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def draw_primitive_inside_2d(plot, tsd, coord, title='', s=1):
    tsd = check_and_reduce_batch(tsd, None, [2, 3])
    sgn = model_utils.convert_tsd_range_to_zero_to_one(tsd).sum(0).view(-1)
    sgn_numpy = tensor2numpy(sgn)

    coord = check_and_reduce_batch(coord, 2, [2, 3])

    coord_numpy = tensor2numpy(coord)

    x = coord_numpy[:, 0]
    y = coord_numpy[:, 1]
    plot.set_aspect('equal')
    plot.scatter(x[sgn_numpy > 0.5], y[sgn_numpy > 0.5], c=[1, 0, 0], s=s)


def draw_primitive_inside_3d(tsd, coord, title='', s=1, th=0.5):
    tsd = check_and_reduce_batch(tsd, None, [2, 3])
    sgn = model_utils.convert_tsd_range_to_zero_to_one(tsd).sum(0).view(-1)
    sgn_numpy = tensor2numpy(sgn)

    coord = check_and_reduce_batch(coord, 3, [2, 3])

    coord_numpy = tensor2numpy(coord)

    x = coord_numpy[:, 0][sgn_numpy > th]
    y = coord_numpy[:, 1][sgn_numpy > th]
    z = coord_numpy[:, 2][sgn_numpy > th]

    plots = []
    marker_opt = dict(size=s)

    plots.append(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                              marker=marker_opt))
    fig = go.Figure(data=plots)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
