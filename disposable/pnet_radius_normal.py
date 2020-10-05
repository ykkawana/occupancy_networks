# %%
import trimesh
import os
import pandas
import pickle
import subprocess
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import tempfile
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
sys.path.insert(
    0,
    '/home/mil/kawana/workspace/occupancy_networks/external/periodic_shapes')
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/atlasnetv2')
from im2mesh.utils import binvox_rw
import math
import numpy as np
import matplotlib.pyplot as plt
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
import eval_utils
from tqdm import tqdm
import yaml
import pickle
from bspnet import utils as bsp_utils
from bspnet import modelSVR
from datetime import datetime
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# %%
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/gen_config_f60k_20200511_163104.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/gen_config_f60k_20200511_163104.yaml'
base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018'
# %%
synset_to_label = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "loudspeaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "vessel"
}
label_to_synset = {v: k for k, v in synset_to_label.items()}

# %%
cfg = config.load_config(config_path, 'configs/default.yaml')
cfg['data']['is_normal_icosahedron'] = True
cfg['data']['icosahedron_subdiv'] = 4
cfg['model']['decoder_kwargs']['is_simpler_sgn'] = True
#cfg['data']['is_normal_uv_sphere'] = True
#cfg['data']['uv_sphere_length'] = 20
cfg['model']['decoder_kwargs']['is_get_radius_direction_as_normal'] = True
assert cfg['method'] == 'pnet'
#eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = True
device = torch.device("cuda:7" if is_cuda else "cpu")

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(base_eval_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          num_workers=4,
                                          shuffle=False)

# Generate
model.eval()

# %%
indices = {}
for idx in range(len(dataset)):
    model_dict = dataset.get_model_dict(idx)
    model_id = model_dict['model']
    class_id = model_dict.get('category', 'n/a')
    indices[(class_id, model_id)] = idx
# %%
target_class = 'car'
target_model_id = 'cbe2dc469c47bb80425b2c354eccabaf'
idx = indices[(label_to_synset[target_class], target_model_id)]
data = dataset[idx]

normal_angles = data.get('angles.normal_angles').to(device).unsqueeze(0)
normal_angles.requires_grad = True
normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
inputs = data.get('inputs').to(device).unsqueeze(0)
feature = model.encode_inputs(inputs)

kwargs = {}
point_scale = cfg['trainer']['pnet_point_scale']

#scaled_coord = points * point_scale
#with torch.no_grad():

EPS = 0.1
#normal_angles = torch.where(normal_angles == 0, normal_angles + torch.tensor([1e-7]).to(device), normal_angles)
"""
normal_angles[:, :, 0] = normal_angles[:, :, 0].clamp(min=(-np.pi + EPS),
                                                      max=(np.pi - EPS))
normal_angles[:, :, 1] = normal_angles[:, :, 1].clamp(min=(-np.pi / 2 + EPS),
                                                      max=(np.pi / 2 - EPS))
"""
output = model.decode(None, None, feature, angles=normal_angles, **kwargs)
normal_vertices, mask, _, sgn_N_NP, radius = output

#uv = torch.tensor(normal_angles, requires_grad=True)
surface_verts = normal_vertices.view(1, -1, 3)[:, mask.view(-1), :]
surface_verts.retain_grad = True

#uv_normals = surface_verts.grad
nparts = normal_vertices.shape[1]
diag_idx = []
for i in range(nparts):
    diag_idx.append(i + i * nparts)
"""
sgn = sgn_N_NP.view(1, nparts**2, -1)[:, diag_idx, :]
#normal_angles.retain_grad = True
uv_normals = -torch.autograd.grad(
    sgn,
    normal_vertices,
    torch.ones_like(sgn),
)[0]
uv_normals = uv_normals / torch.norm(uv_normals, dim=-1,
                                     keepdim=True).clamp(min=1e-7)
"""
thetags = []
phigs = []
theta = normal_angles[..., 0]
phi = normal_angles[..., 1]
for idx in range(nparts):
    r = radius[:, idx, :]
    rg = torch.autograd.grad(
        r,
        normal_angles,
        torch.ones_like(r),
        retain_graph=True,
    )[0]
    thetag = torch.stack([
        rg[..., 0] * theta.cos() * phi.cos() - theta.sin() * phi.cos() * r,
        rg[..., 0] * theta.sin() * phi.cos() + theta.cos() * phi.cos() * r,
        rg[..., 0] * phi.sin()
    ],
                         axis=-1)
    thetags.append(thetag)
    phig = torch.stack([
        rg[..., 1] * theta.cos() * phi.cos() - theta.cos() * phi.sin() * r,
        rg[..., 1] * theta.sin() * phi.cos() - theta.sin() * phi.sin() * r,
        rg[..., 1] * phi.sin() + phi.cos() * r
    ],
                       axis=-1)
    phigs.append(phig)
thetags = torch.cat(thetags, axis=1)
phigs = torch.cat(phigs, axis=1)
uv_normals = torch.cross(thetags, phigs)
print(uv_normals.shape)
#create_graph=True,
#retain_graph=True,
#only_inputs=True,
#allow_unused=True)

verts = (normal_vertices[0, ...] / point_scale).to('cpu').detach().numpy()
#verts = verts[5, :][np.newaxis, ...]
npart, verts_n, _ = verts.shape
verts_all = verts.reshape([-1, 3])
faces = normal_faces[0, ...].to('cpu').detach().numpy()
faces_all = np.concatenate([faces + verts_n * i for i in range(npart)])

# For primitive wise rendering
mesh = trimesh.Trimesh(verts_all, faces_all, process=False)
gt_normals = mesh.vertex_normals


# %%
def normals2rgb(normals, append_dummpy_alpha=False):
    normals = normals / np.clip(
        (normals**2).sum(-1, keepdims=True), 1e-7, None)
    RGB = np.zeros_like(normals)
    x = normals[:, 0]
    y = normals[:, 1]
    z = normals[:, 2]
    absx = np.abs(x)
    absy = np.abs(y)
    absz = np.abs(z)
    leftright = (absy >= absx) & (absy >= absz)
    RGB[leftright, 0] = (1 / absy[leftright]) * x[leftright]
    RGB[leftright, 2] = (1 / absy[leftright]) * z[leftright]
    RGB[leftright & (y > 0), 1] = 1
    RGB[leftright & (y < 0), 1] = -1
    frontback = (absx >= absy) & (absx >= absz)
    RGB[frontback, 1] = (1 / absx[frontback]) * y[frontback]
    RGB[frontback, 2] = (1 / absx[frontback]) * z[frontback]
    RGB[frontback & (x > 0), 0] = 1
    RGB[frontback & (x < 0), 0] = -1
    topbottom = (absz >= absx) & (absz >= absy)
    RGB[topbottom, 0] = (1 / absz[topbottom]) * x[topbottom]
    RGB[topbottom, 1] = (1 / absz[topbottom]) * y[topbottom]
    RGB[topbottom & (z > 0), 2] = 1
    RGB[topbottom & (z < 0), 2] = -1
    RGB = 0.5 * RGB + 0.5
    RGB = np.clip(RGB, 0, 1)
    RGB[np.any(np.isnan(RGB), axis=1), :] = [0., 0., 0.]
    if append_dummpy_alpha:
        RGB = np.concatenate(
            [RGB, np.zeros([RGB.shape[0], 1], dtype=RGB.dtype)], axis=1)
    #RGB[np.isnan(normals),2),:)=nan; % zero vectors are mapped to black
    return RGB


# %%
colors = normals2rgb(uv_normals[0, :,
                                ...].detach().cpu().numpy().reshape(-1, 3),
                     append_dummpy_alpha=False)
#colors = normals2rgb(normals[0, 3, ...].detach().cpu().numpy().reshape(-1, 3),
#                     append_dummpy_alpha=False)
#colors = normals2rgb(gt_normals.reshape(-1, 3), append_dummpy_alpha=False)
#mesh = trimesh.creation.icosphere(subdivisions=4)
mesh.visual.vertex_colors = colors
mesh.show()

# %%
