import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (compute_iou, make_3d_grid)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from im2mesh.atlasnetv2 import utils as atv2_utils
from atlasnetv2.extension import dist_chamfer
import wandb


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
    def __init__(
        self,
        model,
        optimizer,
        device=None,
        input_type='img',
        vis_dir=None,
        threshold=0.5,
        point_scale=1,
        is_pykeops_loss=True,
        debugged=False,
        training_step=None,  # if training_step == None, then directry training model wiht svr. Otherwise 'autoencoder' or 'svr'
        eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.point_scale = point_scale
        self.is_pykeops_loss = is_pykeops_loss
        self.debugged = debugged
        self.training_step = training_step

        self.distChamferL2 = dist_chamfer.chamferDist()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

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
        if 'pointcloud.atlasnetv2_points' in data:
            pointcloud = data.get('pointcloud.atlasnetv2_points').to(
                device) * self.point_scale
        else:
            pointcloud = data.get('pointcloud').to(device) * self.point_scale

        patch = data.get('patch').to(device)
        inputs = data.get('inputs', torch.empty(
            pointcloud.size(0),
            0)).to(device) * (1 if self.debugged else self.point_scale)
        assert self.debugged

        if inputs.ndim == 3:  #pointcloud
            inputs = inputs * self.point_scale

        if self.training_step == 'autoencoder':
            feature = self.model.encode_pointcloud_inputs(pointcloud)
        else:
            feature = self.model.encode_inputs(inputs)

        eval_dict = {}
        kwargs = {}
        with torch.no_grad():
            # General points
            coords = self.model.decode(pointcloud,
                                       None,
                                       feature,
                                       grid=patch,
                                       **kwargs)
            B, N, P, dims = coords.shape
            loss = atv2_utils.chamfer_loss(coords.view(B, N * P, dims),
                                           pointcloud)

        eval_dict['cd1'] = loss.cpu().numpy()

        return eval_dict

    def visualize(self, data, it=0., epoch_it=0.):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        if 'pointcloud.atlasnetv2_points' in data:
            pointcloud = data.get('pointcloud.atlasnetv2_points').to(
                device) * self.point_scale
        else:
            pointcloud = data.get('pointcloud').to(device) * self.point_scale

        patch = data.get('patch').to(device)
        inputs = data.get('inputs', torch.empty(
            pointcloud.size(0),
            0)).to(device) * (1 if self.debugged else self.point_scale)

        if self.training_step == 'autoencoder':
            feature = self.model.encode_pointcloud_inputs(pointcloud)
        else:
            feature = self.model.encode_inputs(inputs)

        eval_dict = {}
        kwargs = {}
        with torch.no_grad():
            # General points
            coords = self.model.decode(pointcloud,
                                       None,
                                       feature,
                                       grid=patch,
                                       **kwargs)
            B, N, P, dims = coords.shape

        input_images = []
        voxels_images = []
        for i in trange(B):
            if not inputs.ndim == 1:  # no input image
                input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
                plot = vis.visualize_data(inputs[i].cpu() / self.point_scale,
                                          self.input_type,
                                          input_img_path,
                                          return_plot=True)
                input_images.append(
                    wandb.Image(plot, caption='input image {}'.format(i)))
            plot = vis.visualize_pointcloud(
                coords[i].cpu().view(N * P, dims) / self.point_scale,
                normals=None,
                out_file=os.path.join(self.vis_dir, '%03d.png' % i),
                return_plot=True)
            voxels_images.append(
                wandb.Image(plot, caption='voxel {}'.format(i)))
        if not inputs.ndim == 1:  # no input image
            wandb.log({'input_image': input_images}, step=it)
        wandb.log({'pointcloud_visualization': voxels_images}, step=it)

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        assert self.debugged
        device = self.device
        if 'pointcloud.atlasnetv2_points' in data:
            pointcloud = data.get('pointcloud.atlasnetv2_points').to(
                device) * self.point_scale
        else:
            pointcloud = data.get('pointcloud').to(device) * self.point_scale

        patch = data.get('patch').to(device)
        inputs = data.get('inputs', torch.empty(
            pointcloud.size(0),
            0)).to(device) * (1 if self.debugged else self.point_scale)
        assert self.debugged
        kwargs = {}

        feature = self.model.encode_inputs(inputs)
        if self.training_step in ['autoencoder', 'svr']:
            pointcloud_feature = self.model.encode_pointcloud_inputs(
                pointcloud)

        # General points
        if self.train_step == 'svr':
            loss = torch.mean((feature - pointcloud_feature.detach())**2)
        else:
            coords = self.model.decode(
                pointcloud,
                None,
                feature if self.training_step is None else pointcloud_feature,
                grid=patch,
                **kwargs)
            B, N, P, dims = coords.shape

            if self.is_pykeops_loss:
                loss = atv2_utils.chamfer_loss(coords.view(B, N * P, dims),
                                               pointcloud)
            else:
                dist1, dist2 = self.distChamferL2(coords.view(B, N * P, dims),
                                                  pointcloud)
                loss = torch.mean(dist1) + torch.mean(dist2)

        losses = {'total_loss': loss}
        return losses
