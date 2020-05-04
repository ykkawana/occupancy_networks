# %%
import argparse
import os
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
import sys
sys.path.insert(
    0,
    '/home/mil/kawana/workspace/occupancy_networks/external/periodic_shapes')
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/atlasnetv2')
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/bspnet')
import subprocess
import hashlib
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import shutil
import yaml
from collections import OrderedDict
import eval_utils
import random
from bspnet.bspt import digest_bsp, get_mesh, get_mesh_watertight
import plotly.graph_objects as go
import tempfile
import torch
from kaolin.transforms import pointcloudfunc as pcfunc
from im2mesh.atlasnetv2 import utils
import time
import trimesh

# %%


def realign(src, src_ref, tgt, th=0.001, iters=200):
    EPS = 1e-12
    #th = 0.1 # mode = l1
    mode = 'l2'
    lr = 1e-2
    optim = torch.optim.RMSprop

    assert src.ndim == src_ref.ndim == tgt.ndim == 3
    # Compute the relative scaling factor and scale the src cloud.
    src_min = src_ref.min(axis=-2, keepdims=True)[0]
    src_max = src_ref.max(axis=-2, keepdims=True)[0]
    tgt_min = tgt.min(axis=-2, keepdims=True)[0]
    tgt_max = tgt.max(axis=-2, keepdims=True)[0]

    #src = ((src - src_min) /
    #       (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min

    scaled_src_ref = (
        (src_ref - src_min) /
        (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min

    cont_src_ref = scaled_src_ref.contiguous()
    cont_tgt = tgt.contiguous()
    loss = utils.chamfer_loss(cont_src_ref, cont_tgt, mode=mode)
    print('cd', loss.item())
    if loss < th:
        src = ((src - src_min) /
               (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min
        return src
    else:
        scale = torch.nn.Parameter(
            torch.ones([1, 1, 3],
                       device=src.device,
                       dtype=src.dtype,
                       requires_grad=True))
        shift = torch.nn.Parameter(
            torch.zeros([1, 1, 3],
                        device=src.device,
                        dtype=src.dtype,
                        requires_grad=True))

        src_ref = src_ref.contiguous()
        src_ref.requires_grad = False
        tgt = tgt.contiguous()
        tgt.requires_grad = False

        print('scale', scale.requires_grad, 'shift', shift.requires_grad,
              'src', src.requires_grad, 'src_ref', src_ref.requires_grad,
              'tgt', tgt.requires_grad)

        src_min = src_ref.min(axis=-2, keepdims=True)[0]
        src_max = src_ref.max(axis=-2, keepdims=True)[0]
        tgt_min = tgt.min(axis=-2, keepdims=True)[0]
        tgt_max = tgt.max(axis=-2, keepdims=True)[0]

        src_min_c = src_min.close()
        src_max_c = src_max.close()
        tgt_min_c = tgt_min.close()
        tgt_max_c = tgt_max.close()

        optimizer = optim([scale, shift], lr=lr)
        #optimizer = torch.optim.RMSprop([scale, shift], lr=1e-2)
        for it in range(iters):
            optimizer.zero_grad()
            scaled_src_ref = ((src_ref - src_min + shift) /
                              (src_max - src_min + EPS)) * scale * (
                                  tgt_max - tgt_min) + tgt_min
            loss = utils.chamfer_loss(scaled_src_ref, tgt, mode=mode)
            if loss < th:
                break
            loss.backward()
            optimizer.step()
        print('final cd', loss.item(), it, scale, shift)
        src = (
            (src - src_min + shift) /
            (src_max - src_min + EPS)) * scale * (tgt_max - tgt_min) + tgt_min
        return src.detach()


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--dontcopy', action='store_true', help='Do not use cuda.')
parser.add_argument('--no_copy_but_create_new',
                    action='store_true',
                    help='Do not use cuda.')

# Get configuration and basic arguments
args, unknown_args = parser.parse_known_args()
args.config = '/home/mil/kawana/workspace/occupancy_networks/configs/img/debug_bspnet.yaml'
cfg = config.load_config(args.config, 'configs/default.yaml')

eval_utils.update_dict_with_options(cfg, unknown_args)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands

dataset = config.get_dataset('test', cfg, return_idx=True)
model = config.get_model(cfg, device=device, dataset=dataset)

out_dir = cfg['training']['out_dir']
checkpoint_io = CheckpointIO(out_dir, model=model)
try:
    checkpoint_io.load(cfg['test']['model_file'])
except FileExistsError:
    print('Model file does not exist. Exiting.')
    exit()

# Trainer
trainer = config.get_trainer(model, None, cfg, device=device)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

# Evaluate
model.eval()

eval_dicts = []
print('Evaluating networks...')
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=data.collate_remove_none,
                                          worker_init_fn=data.worker_init_fn)

# %%
deg = 90.
rad = deg / 180. * np.pi
rotz = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad),
                                         np.cos(rad), 0], [0., 0., 1.]]
roty = [[np.cos(rad), 0., np.sin(rad)], [0., 1., 0.],
        [-np.sin(rad), 0., np.cos(rad)]]
zrotm = torch.tensor(roty, device=device).float()
data = dataset[1]

inputs = data.get('inputs').to(device).unsqueeze(0)

points_iou = torch.from_numpy(data.get('points_iou')).to(device).unsqueeze(0)
points_iou = pcfunc.rotate(points_iou, zrotm)
occ_iou = torch.from_numpy(data.get('points_iou.occ')).to(device).unsqueeze(0)

occ_pointcloud = torch.from_numpy(
    data.get('pointcloud')).to(device).unsqueeze(0)
occ_pointcloud = pcfunc.rotate(occ_pointcloud, zrotm)
imnet_pointcloud = data.get('pointcloud.imnet_points').to(
    device).float().unsqueeze(0)

t0 = time.time()
scaled_points_iou = realign(points_iou, occ_pointcloud, imnet_pointcloud)
print('realign takes:', time.time() - t0)
scaled_occ_pointcloud = realign(occ_pointcloud, occ_pointcloud,
                                imnet_pointcloud)
#scaled_occ_pointcloud = occ_pointcloud

with torch.no_grad():
    one = torch.ones([1], device=device,
                     dtype=scaled_points_iou.dtype).float().view(
                         1, 1, 1).expand(points_iou.shape[0],
                                         points_iou.shape[1], -1)
    input_points_iou = torch.cat([scaled_points_iou, one], axis=2)
    _, out_m, _, _ = model(inputs, None, None, None, is_training=False)

    _, _, _, logits = model(None,
                            None,
                            out_m,
                            input_points_iou,
                            is_training=False)

occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
occ_iou_hat_np = ((1 - logits).clamp(min=0, max=1) >= 0.99).cpu().numpy()
#occ_iou_hat_np = (logits >= 0.5).cpu().numpy()

select = random.sample(range(imnet_pointcloud.shape[1]), 1000)
imnet_pcd_np = imnet_pointcloud[0].to('cpu').numpy()[select, :]

select = random.sample(range(occ_iou_np.sum()), min(1000, occ_iou_np.sum()))
points_plot = scaled_points_iou[0].to('cpu').numpy()[occ_iou_np[0], :][
    select, :]

select = random.sample(range(scaled_occ_pointcloud.shape[1]), 1000)
occ_pcd_np = scaled_occ_pointcloud[0].to('cpu').numpy()[select, :]

select = random.sample(range(occ_iou_hat_np.sum()),
                       min(1000, occ_iou_hat_np.sum()))
logits_points_plot = scaled_points_iou[0].to('cpu').numpy()[occ_iou_hat_np[
    0, :, 0], :][select, :]

plots = []
marker_opt = dict(size=1)
plots.append(
    go.Scatter3d(x=imnet_pcd_np[:, 0],
                 y=imnet_pcd_np[:, 1],
                 z=imnet_pcd_np[:, 2],
                 mode='markers',
                 name='imnet pcd',
                 marker=marker_opt))
plots.append(
    go.Scatter3d(x=points_plot[:, 0],
                 y=points_plot[:, 1],
                 z=points_plot[:, 2],
                 mode='markers',
                 name='occ',
                 marker=marker_opt))
plots.append(
    go.Scatter3d(x=occ_pcd_np[:, 0],
                 y=occ_pcd_np[:, 1],
                 z=occ_pcd_np[:, 2],
                 mode='markers',
                 name='occ pcd',
                 marker=marker_opt))
"""
plots.append(
    go.Scatter3d(x=logits_points_plot[:, 0],
                 y=logits_points_plot[:, 1],
                 z=logits_points_plot[:, 2],
                 mode='markers',
                 name='logits occ',
                 marker=marker_opt))
"""
fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()
# %%
deg = -90.
rad = deg / 180. * np.pi
rotz = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad),
                                         np.cos(rad), 0], [0., 0., 1.]]
roty = [[np.cos(rad), 0., np.sin(rad)], [0., 1., 0.],
        [-np.sin(rad), 0., np.cos(rad)]]
zrotm = torch.tensor(roty, device=device).float()
occ_pointcloud = torch.from_numpy(
    data.get('pointcloud')).to(device).unsqueeze(0)
imnet_pointcloud = data.get('pointcloud.imnet_points').to(
    device).float().unsqueeze(0)
imnet_pointcloud = pcfunc.rotate(imnet_pointcloud, zrotm)

t0 = time.time()
scaled_imnet_pointcloud = realign(imnet_pointcloud, imnet_pointcloud,
                                  occ_pointcloud)
print('realign takes:', time.time() - t0)

select = random.sample(range(imnet_pointcloud.shape[1]), 1000)
imnet_pcd_np = scaled_imnet_pointcloud[0].to('cpu').numpy()[select, :]
select = random.sample(range(occ_pointcloud.shape[1]), 1000)
occ_pcd_np = occ_pointcloud[0].to('cpu').numpy()[select, :]

mesh = trimesh.load(
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_164210/generation_20200501_164653/meshes/03636649/e6629a35985260b8702476de6c89c9e9.off'
)
mesh_points = mesh.sample(1000)
surf_points = np.load(
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_164210/generation_20200501_172142/meshes/03636649/e6629a35985260b8702476de6c89c9e9_vertex_attributes.npz'
)['vertices']
#np.random.shuffle(surf_points)
surf_points = surf_points[select, :]

plots = []
plots.append(
    go.Scatter3d(x=occ_pcd_np[:, 0],
                 y=occ_pcd_np[:, 1],
                 z=occ_pcd_np[:, 2],
                 mode='markers',
                 name='occ pcd',
                 marker=marker_opt))
plots.append(
    go.Scatter3d(x=imnet_pcd_np[:, 0],
                 y=imnet_pcd_np[:, 1],
                 z=imnet_pcd_np[:, 2],
                 mode='markers',
                 name='imnet pcd',
                 marker=marker_opt))
"""
plots.append(
    go.Scatter3d(x=mesh_points[:, 0],
                 y=mesh_points[:, 1],
                 z=mesh_points[:, 2],
                 mode='markers',
                 name='mesh sampled points',
                 marker=marker_opt))
"""
plots.append(
    go.Scatter3d(x=surf_points[:, 0],
                 y=surf_points[:, 1],
                 z=surf_points[:, 2],
                 mode='markers',
                 name='surf points',
                 marker=marker_opt))
fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()
