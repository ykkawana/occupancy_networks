from periodic_shapes.losses import custom_chamfer_loss
import torch
import numpy as np


def test_custom_chamfer_loss_pykeops():
    torch.autograd.set_detect_anomaly(True)
    points_n = 100
    sampled_points_n = 200
    pred_points = torch.nn.Parameter(torch.Tensor(1, 6, points_n, 3))
    torch.nn.init.uniform_(pred_points)
    sampled_points = torch.rand([1, sampled_points_n, 3]).float()
    pred_mask = torch.ones([1, 6, points_n]).float()

    loss = custom_chamfer_loss.custom_chamfer_loss(pred_points,
                                                   sampled_points,
                                                   surface_mask=pred_mask,
                                                   pykeops=False)

    loss_pykeops = custom_chamfer_loss.custom_chamfer_loss(
        pred_points, sampled_points, surface_mask=pred_mask, pykeops=True)

    assert torch.all(torch.eq(loss, loss_pykeops))

    loss_pykeops.backward()


def test_custom_chamfer_loss_apply_mask_as_inf_dist():
    torch.autograd.set_detect_anomaly(True)
    points_n = 100
    sampled_points_n = 200
    pred_points = torch.nn.Parameter(torch.Tensor(1, 6, points_n, 3))
    torch.nn.init.uniform_(pred_points)
    sampled_points = torch.rand([1, sampled_points_n, 3]).float()
    pred_mask = torch.ones([1, 6, points_n]).float()

    loss = custom_chamfer_loss.custom_chamfer_loss(
        pred_points,
        sampled_points,
        surface_mask=pred_mask,
        pykeops=False,
        apply_surface_mask_before_chamfer=True)

    loss_pykeops = custom_chamfer_loss.custom_chamfer_loss(
        pred_points,
        sampled_points,
        surface_mask=pred_mask,
        pykeops=True,
        apply_surface_mask_before_chamfer=True)

    assert torch.all(torch.eq(loss, loss_pykeops))

    pred_mask = torch.ones([1, 6, points_n]).float()

    pred_mask[:, :, 10] = 0.
    pred_mask[:, :, 50] = 0.

    loss = custom_chamfer_loss.custom_chamfer_loss(
        pred_points,
        sampled_points,
        surface_mask=pred_mask,
        pykeops=False,
        apply_surface_mask_before_chamfer=True)

    loss_pykeops = custom_chamfer_loss.custom_chamfer_loss(
        pred_points,
        sampled_points,
        surface_mask=pred_mask,
        pykeops=True,
        apply_surface_mask_before_chamfer=True)

    loss_pykeops2 = custom_chamfer_loss.custom_chamfer_loss(
        pred_points,
        sampled_points,
        surface_mask=pred_mask,
        pykeops=True,
        apply_surface_mask_before_chamfer=False)

    assert torch.all(torch.eq(loss, loss_pykeops))
    assert not torch.all(torch.eq(loss_pykeops2, loss_pykeops))
