import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (compute_iou, make_3d_grid)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from periodic_shapes.models import model_utils
from periodic_shapes.losses import custom_chamfer_loss
import wandb
from kaolin import rep


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''
    def __init__(self,
                 model,
                 optimizer,
                 device=None,
                 input_type='img',
                 vis_dir=None,
                 threshold=0.5,
                 eval_sample=False,
                 overlap_reg_coef=1.,
                 self_overlap_reg_coef=1.,
                 chamfer_loss_coef=1.,
                 occupancy_loss_coef=1.,
                 normal_loss_coef=1.,
                 pnet_point_scale=6.,
                 overlap_reg_threshold=1.2,
                 self_overlap_reg_threshold=0.1,
                 add_pointcloud_occ=False,
                 is_logits_by_max=False,
                 is_strict_chamfer=False,
                 is_logits_by_sign_filter=False,
                 is_normal_loss=False,
                 is_onet_style_occ_loss=False,
                 is_logits_by_softmax=False,
                 is_l2_occ_loss=False,
                 sgn_scale=100,
                 is_sdf=False,
                 is_logits_by_logsumexp=False,
                 is_logits_by_min=False,
                 is_cvx_net_merged_loss=False,
                 cvx_net_merged_loss_topk_samples=10,
                 cvx_net_merged_loss_coef=1,
                 is_eval_logits_by_max=False,
                 is_radius_reg=False,
                 radius_reg_coef=1.,
                 use_surface_mask=True,
                 sgn_offset=0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        self.overlap_reg_coef = overlap_reg_coef
        self.self_overlap_reg_coef = self_overlap_reg_coef
        self.chamfer_loss_coef = chamfer_loss_coef
        self.occupancy_loss_coef = occupancy_loss_coef
        self.normal_loss_coef = normal_loss_coef
        self.pnet_point_scale = pnet_point_scale
        self.overlap_reg_threshold = overlap_reg_threshold
        self.self_overlap_reg_threshold = self_overlap_reg_threshold
        self.add_pointcloud_occ = add_pointcloud_occ
        self.is_logits_by_max = is_logits_by_max
        self.is_strict_chamfer = is_strict_chamfer
        self.is_logits_by_sign_filter = is_logits_by_sign_filter
        self.is_normal_loss = is_normal_loss
        self.is_onet_style_occ_loss = is_onet_style_occ_loss
        self.is_logits_by_softmax = is_logits_by_softmax
        self.is_l2_occ_loss = is_l2_occ_loss
        self.sgn_scale = sgn_scale
        self.is_sdf = is_sdf
        self.is_logits_by_logsumexp = is_logits_by_logsumexp
        self.is_logits_by_min = is_logits_by_min
        self.is_eval_logits_by_max = is_eval_logits_by_max
        self.is_cvx_net_merged_loss = is_cvx_net_merged_loss
        self.cvx_net_merged_loss_topk_samples = cvx_net_merged_loss_topk_samples
        self.cvx_net_merged_loss_coef = cvx_net_merged_loss_coef
        self.sgn_offset = sgn_offset
        self.is_radius_reg = is_radius_reg
        self.radius_reg_coef = radius_reg_coef
        self.use_surface_mask = use_surface_mask

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if self.add_pointcloud_occ:
            assert not self.is_sdf

        if self.is_eval_logits_by_max:
            assert not self.is_sdf

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.compute_loss(data)
        losses['total_loss'].backward()
        self.optimizer.step()
        return losses

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        angles = data.get('angles').to(device)

        kwargs = {}
        """
        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()
        """

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            _, _, sgn, _, _ = self.model(points_iou * self.pnet_point_scale,
                                         inputs,
                                         sample=self.eval_sample,
                                         angles=angles,
                                         **kwargs)
        if self.is_sdf:
            if self.is_logits_by_min:
                logits = (sgn.min(1)[0] <= 0).float()
            else:
                raise NotImplementedError
        else:
            if self.is_eval_logits_by_max:
                logits = (sgn >= 0.).float().max(1)[0]
            elif self.is_logits_by_max:
                logits = model_utils.convert_tsd_range_to_zero_to_one(
                    sgn.max(1)[0])
            elif self.is_logits_by_sign_filter:
                positive = torch.relu(sgn).sum(1)
                negative = torch.relu(-sgn).sum(1)
                logits = torch.where(positive >= negative, positive, -negative)
            else:
                logits = model_utils.convert_tsd_range_to_zero_to_one(sgn).sum(
                    1)

        occ_iou_np = (occ_iou >= self.threshold).cpu().numpy()
        occ_iou_hat_np = (logits >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid((-0.5 + 1 / 64, ) * 3,
                                         (0.5 - 1 / 64, ) * 3, (32, ) * 3)
            points_voxels = points_voxels.expand(batch_size,
                                                 *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                _, _, sgn, _, _ = self.model(points_voxels *
                                             self.pnet_point_scale,
                                             inputs,
                                             sample=self.eval_sample,
                                             angles=angles,
                                             **kwargs)

            if self.is_sdf:
                if self.is_logits_by_min:
                    logits = (sgn.min(1)[0] <= 0).float()
                else:
                    raise NotImplementedError
            else:
                if self.is_eval_logits_by_max:
                    logits = (sgn >= 0.).float().max(1)[0]
                elif self.is_logits_by_max:
                    logits = model_utils.convert_tsd_range_to_zero_to_one(
                        sgn.max(1)[0])
                elif self.is_logits_by_sign_filter:
                    positive = torch.relu(sgn).sum(1)
                    negative = torch.relu(-sgn).sum(1)
                    logits = torch.where(positive >= negative, positive,
                                         -negative)
                else:
                    logits = model_utils.convert_tsd_range_to_zero_to_one(
                        sgn).sum(1)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (logits >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data, it=0., epoch_it=0.):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)
        angles = data.get('angles').to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            _, _, sgn, _, _ = self.model(p * self.pnet_point_scale,
                                         inputs,
                                         sample=self.eval_sample,
                                         angles=angles,
                                         **kwargs)

        if self.is_sdf:
            if self.is_logits_by_min:
                logits = (sgn.min(1)[0] <= 0).float()
            else:
                raise NotImplementedError
        else:
            if self.is_logits_by_max:
                logits = model_utils.convert_tsd_range_to_zero_to_one(
                    sgn.max(1)[0])
            elif self.is_logits_by_sign_filter:
                positive = torch.relu(sgn).sum(1)
                negative = torch.relu(-sgn).sum(1)
                logits = torch.where(positive >= negative, positive, -negative)
            else:
                logits = model_utils.convert_tsd_range_to_zero_to_one(sgn).sum(
                    1)
        occ_hat = logits.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        input_images = []
        voxels_images = []
        for i in trange(batch_size):
            if not inputs.ndim == 1:  # no input image
                input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
                plot = vis.visualize_data(inputs[i].cpu(),
                                          self.input_type,
                                          input_img_path,
                                          return_plot=True)
                input_images.append(
                    wandb.Image(plot, caption='input image {}'.format(i)))
            plot = vis.visualize_voxels(voxels_out[i],
                                        os.path.join(self.vis_dir,
                                                     '%03d.png' % i),
                                        return_plot=True)
            voxels_images.append(
                wandb.Image(plot, caption='voxel {}'.format(i)))
        if not inputs.ndim == 1:  # no input image
            wandb.log({'input_image': input_images}, step=it)
        wandb.log({'voxel_visualization': voxels_images}, step=it)

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        if self.is_sdf:
            points = data.get('sdf_points').to(device)
            sdf = data.get('sdf_points.distances').to(device)
            occ = (sdf <= 0).float()
        else:
            points = data.get('points').to(device)
            occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        angles = data.get('angles').to(device)
        pointcloud = data['pointcloud'].to(device)

        if self.is_normal_loss:
            target_normals = data['pointcloud.normals'].to(device)
            normal_faces = data.get('angles.normal_face').to(device)
            normal_angles = data.get('angles.normal_angles').to(device)

        kwargs = {}
        if self.add_pointcloud_occ and not self.is_sdf:
            points = torch.cat([points, pointcloud], axis=1)
            occ = torch.cat([
                occ,
                torch.ones(
                    pointcloud.shape[:2], device=occ.device, dtype=occ.dtype)
            ],
                            axis=1)
        """
        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(points, occ, c, **kwargs)
        z = q_z.rsample()

        # General points

        output = self.model.decode(scaled_coord,
                                          z,
                                          c,
                                          angles=angles,
                                          **kwargs)

        """
        scaled_coord = points * self.pnet_point_scale
        output = self.model(scaled_coord,
                            inputs,
                            sample=True,
                            angles=angles,
                            **kwargs)

        super_shape_point, surface_mask, sgn, sgn_BxNxNP, radius = output

        # losses
        if self.is_sdf:
            if self.is_logits_by_logsumexp:
                raise NotImplementedError
            if self.is_logits_by_sign_filter:
                raise NotImplementedError
            if self.is_logits_by_softmax:
                raise NotImplementedError
            if self.is_logits_by_max:
                raise NotImplementedError
            if self.is_logits_by_min:
                logits = sgn.min(1)[0]
            else:
                raise NotImplementedError
            """
            delta = 0.01 * self.pnet_point_scale**2
            occupancy_loss = (logits.clamp(min=-delta, max=delta) -
                              (sdf * self.pnet_point_scale**2).clamp(
                                  min=-delta, max=delta)).abs().mean()
            """
            occupancy_loss = (logits -
                              (sdf * self.pnet_point_scale**2)).abs().mean()

        else:
            if self.is_logits_by_max:
                logits = sgn.max(1)[0]
            elif self.is_logits_by_sign_filter:
                positive = torch.relu(sgn).sum(1)
                negative = torch.relu(-sgn).sum(1)
                logits = torch.where(positive >= negative, positive, -negative)
            elif self.is_logits_by_softmax:
                logits = (torch.softmax(sgn * 10, 1) * sgn).sum(1)
            else:
                logits = model_utils.convert_tsd_range_to_zero_to_one(
                    sgn, scale=self.sgn_scale).sum(1)

            if self.is_onet_style_occ_loss:
                loss_i = F.binary_cross_entropy_with_logits(logits,
                                                            occ,
                                                            reduction='none')
                occupancy_loss = loss_i.sum(-1).mean()
            elif self.is_l2_occ_loss:
                occupancy_loss = ((logits - occ)**2).sum()
            else:
                occupancy_loss = F.binary_cross_entropy_with_logits(
                    logits - self.sgn_offset, occ)

        scaled_target_point = pointcloud * self.pnet_point_scale

        chamfer_loss, _ = custom_chamfer_loss.custom_chamfer_loss(
            super_shape_point,
            scaled_target_point,
            surface_mask=(surface_mask if self.use_surface_mask else None),
            prob=None,
            pykeops=True,
            apply_surface_mask_before_chamfer=self.is_strict_chamfer)

        if self.is_normal_loss:
            """
            output = self.model.decode(scaled_coord,
                                       z,
                                       c,
                                       angles=normal_angles,
                                       **kwargs)
            """
            output = self.model(scaled_coord,
                                inputs,
                                sample=True,
                                angles=normal_angles,
                                **kwargs)

            normal_vertices, normal_mask, _, _, _ = output

            B, N, P, D = normal_vertices.shape
            normal_faces_all = torch.cat([(normal_faces + idx * P)
                                          for idx in range(N)],
                                         axis=1)
            normals = vertex_normals(normal_vertices.view(B, -1, D),
                                     normal_faces_all).view(B, N, P, D)
            _, normal_loss = custom_chamfer_loss.custom_chamfer_loss(
                normal_vertices,
                scaled_target_point,
                source_normals=normals,
                target_normals=target_normals,
                surface_mask=normal_mask,
                prob=None,
                pykeops=True,
                apply_surface_mask_before_chamfer=self.is_strict_chamfer)

        # regularizers
        overlap_reg = torch.relu(logits - self.overlap_reg_threshold).mean()
        batch, n_primitives, _ = sgn_BxNxNP.shape
        bnnp_tsd_reshaped = sgn_BxNxNP.view(batch, n_primitives, n_primitives,
                                            -1)[:,
                                                range(n_primitives),
                                                range(n_primitives), :]
        self_overlap_reg = torch.relu(bnnp_tsd_reshaped -
                                      self.self_overlap_reg_threshold).mean()

        total_loss = (occupancy_loss * self.occupancy_loss_coef +
                      chamfer_loss * self.chamfer_loss_coef +
                      overlap_reg * self.overlap_reg_coef +
                      self_overlap_reg * self.self_overlap_reg_coef)

        if self.is_cvx_net_merged_loss:
            merged_loss = torch.topk(sgn,
                                     self.cvx_net_merged_loss_topk_samples,
                                     2)[0]
            merged_loss = (torch.relu(-merged_loss)**
                           2).mean(-1).mean() * self.cvx_net_merged_loss_coef
            print('cvx merged loss:', merged_loss.item())
            total_loss = total_loss + merged_loss

        if self.is_normal_loss:
            total_loss = total_loss + normal_loss * self.normal_loss_coef

        if self.is_radius_reg:
            assert radius is not None
            radius_reg_loss = torch.relu(1. -
                                         radius.std()) * self.radius_reg_coef
            print('radius reg loss', radius_reg_loss.item())
            total_loss = total_loss + radius_reg_loss
        losses = {
            'total_loss': total_loss,
            'occupancy_loss': occupancy_loss * self.occupancy_loss_coef,
            'chamfer_loss': chamfer_loss * self.chamfer_loss_coef,
            'overlap_reg': overlap_reg * self.overlap_reg_coef,
            'self_overlap_reg': self_overlap_reg * self.self_overlap_reg_coef
        }
        if self.is_normal_loss:
            losses.update({'normal_loss': normal_loss * self.normal_loss_coef})
        return losses


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros([bs * nv, 3], dtype=vertices.dtype).to(device)
    faces = faces + (torch.arange(bs, dtype=torch.int32, device=device) *
                     nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0, faces[:, 1].long(),
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                    vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(
        0, faces[:, 2].long(),
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                    vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(
        0, faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                    vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
