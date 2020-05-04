import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (compute_iou, make_3d_grid)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from bspnet import modelSVR
from bspnet import utils
import numpy as np


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
                 eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        self.gen_helper = modelSVR.BSPNetMeshGenerator(self.model,
                                                       device=self.device)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        raise NotImplementedError

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device

        inputs = data.get('inputs').to(device)

        points_iou = data.get('points_iou').to(device)

        points_iou = self.gen_helper.roty90(points_iou)
        occ_iou = data.get('points_iou.occ').to(device)

        occ_pointcloud = data.get('pointcloud').to(device)
        occ_pointcloud = self.gen_helper.roty90(occ_pointcloud)
        imnet_pointcloud = data.get('pointcloud.imnet_points').to(
            device).float()

        scaled_points_iou = utils.realign(points_iou, occ_pointcloud,
                                          imnet_pointcloud)

        with torch.no_grad():
            one = torch.ones([1],
                             device=self.device,
                             dtype=scaled_points_iou.dtype).float().view(
                                 1, 1, 1).expand(points_iou.shape[0],
                                                 points_iou.shape[1], -1)
            input_points_iou = torch.cat([scaled_points_iou, one], axis=2)
            _, _, _, logits = self.model(inputs,
                                         None,
                                         None,
                                         input_points_iou,
                                         is_training=False)

        occ_iou_np = (occ_iou >= self.threshold).cpu().numpy()
        occ_iou_hat_np = ((1 - logits).clamp(min=0, max=1) >=
                          0.99).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()

        eval_dict = {}
        eval_dict['iou'] = iou

        return eval_dict

    def visualize(self, data, it=0., epoch_it=0.):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        raise NotImplementedError

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        raise NotImplementedError


def realign(src, src_ref, tgt):
    EPS = 1e-12
    assert src.ndim == src_ref.ndim == tgt.ndim == 3
    # Compute the relative scaling factor and scale the src cloud.
    src_min = src_ref.min(axis=-2, keepdims=True)[0]
    src_max = src_ref.max(axis=-2, keepdims=True)[0]
    tgt_min = tgt.min(axis=-2, keepdims=True)[0]
    tgt_max = tgt.max(axis=-2, keepdims=True)[0]

    src = ((src - src_min) /
           (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min
    return src
