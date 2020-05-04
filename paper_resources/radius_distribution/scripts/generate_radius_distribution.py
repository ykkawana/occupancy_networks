# %%
import argparse
import torch
import os
import subprocess
import hashlib
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import shutil
import yaml
import matplotlib.pyplot as plt
from collections import OrderedDict
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/atlasnetv2')
sys.path.insert(
    0,
    '/home/mil/kawana/workspace/occupancy_networks/external/periodic_shapes')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

# %%
config_path = '/home/mil/kawana/workspace/occupancy_networks/configs/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg.yaml'
"""
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_oceff10_dense_normal_loss_pointcloud_n_4096_20200414_051415/config.yaml'
"""
resource_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/radius_distribution/resources'

project_base_path = '/home/mil/kawana/workspace/occupancy_networks'
use_cache_flag = False

os.chdir(project_base_path)

# %%
# Get configuration and basic arguments
cfg = config.load_config(config_path, 'configs/default.yaml')
cfg['data']['icosahedron_uv_margin'] = 0
cfg['data']['icosahedron_uv_margin_phi'] = 0
cfg['data']['points_subsample'] = 2
# Enable to get radius
cfg['trainer']['is_radius_reg'] = True
cfg['data']['debug'] = {'sample_n': 10}

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
model = config.get_model(cfg, device=device, dataset=dataset)

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
thetas = np.linspace(-np.pi, np.pi - 2 * np.pi / 100, 100)
phis = np.linspace(-np.pi / 2, np.pi / 2 - np.pi / 100, 100)
theta_radius_list = []
phi_radius_list = []
theta_intra_primitive_radius_list_std = []
theta_intra_primitive_radius_list_mean = []
phi_intra_primitive_radius_list = []
# Handle each dataset separately
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue
    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    points = data.get('points').to(device)
    inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
    theta = torch.cat(torch.meshgrid([
        torch.tensor(thetas, device=device).float(),
        torch.zeros([1], device=device).float()
    ], ),
                      axis=-1).unsqueeze(0)
    phi = torch.cat(torch.meshgrid([
        torch.tensor(phis, device=device).float(),
        torch.zeros([1], device=device).float()
    ], ),
                    axis=-1)[..., [1, 0]].unsqueeze(0)
    points = data.get('points').to(device)

    feature = model.encode_inputs(inputs)

    kwargs = {}
    point_scale = cfg['trainer']['pnet_point_scale']

    scaled_coord = points * point_scale
    output_theta = model.decode(scaled_coord,
                                None,
                                feature,
                                angles=theta,
                                **kwargs)
    _, _, _, _, theta_radius = output_theta
    theta_radius /= point_scale
    theta_radius_list.append(
        theta_radius.mean(axis=(0, 1)).cpu().detach().numpy())
    theta_intra_primitive_radius_list_std.append(
        theta_radius.std(axis=2).mean(axis=(0)).cpu().detach().numpy())
    theta_intra_primitive_radius_list_mean.append(
        theta_radius.mean(axis=2).mean(axis=(0)).cpu().detach().numpy())
    output_phi = model.decode(scaled_coord,
                              None,
                              feature,
                              angles=phi,
                              **kwargs)
    _, _, _, _, phi_radius = output_phi
    phi_radius /= point_scale
    phi_radius_list.append(phi_radius.mean(axis=(0, 1)).cpu().detach().numpy())
    phi_intra_primitive_radius_list.append(
        phi_radius.std(axis=2).mean(axis=(0)).cpu().detach().numpy())
# %%
thetas_in_deg = thetas / np.pi * 180
# Create pandas dataframe and save
thetas_mean_df = pd.DataFrame(theta_radius_list, columns=thetas_in_deg).mean()
thetas_std_df = pd.DataFrame(theta_radius_list, columns=thetas_in_deg).std()

thetas_df = pd.DataFrame()
thetas_df['deg'] = thetas_in_deg
thetas_df['mean'] = thetas_mean_df.values
thetas_df['std'] = thetas_std_df.values
#theta_df.to_pickle('test')

phis_in_deg = phis / np.pi * 180
# Create pandas dataframe and save
phis_mean_df = pd.DataFrame(phi_radius_list, columns=phis_in_deg).mean()
phis_std_df = pd.DataFrame(phi_radius_list, columns=phis_in_deg).std()

phis_df = pd.DataFrame()
phis_df['deg'] = phis_in_deg
phis_df['mean'] = phis_mean_df.values
phis_df['std'] = phis_std_df.values
#theta_df.to_pickle('test')

thetas_intra_df_mean = pd.DataFrame(
    theta_intra_primitive_radius_list_mean).mean()
thetas_intra_df_std = pd.DataFrame(
    theta_intra_primitive_radius_list_std).mean()

thetas_intra_df = pd.DataFrame()
thetas_intra_df['deg'] = np.arange(0, len(thetas_intra_df_mean))
thetas_intra_df['mean'] = thetas_intra_df_mean.values
thetas_intra_df['std'] = thetas_intra_df_std.values
# %%
fig = plt.figure()
degs = thetas_df['deg']
band = thetas_df['std'] * 2
mean = thetas_df['mean']
ax = fig.add_subplot(1, 2, 1)
ax.plot(degs, mean)
ax.fill_between(degs, mean - band, mean + band, color='gray', alpha=0.2)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
ax.legend(['mean', '$2\sigma$'])

degs = phis_df['deg']
band = phis_df['std'] * 2
mean = phis_df['mean']
ax = fig.add_subplot(1, 2, 2)
ax.plot(degs, mean)
ax.fill_between(degs, mean - band, mean + band, color='gray', alpha=0.2)
ax.set_xticks([-90, 0, 90])
ax.set_xticklabels(['$-\pi/2$', '0', '$\pi/2$'])
ax.legend(['mean', '$2\sigma$'])

fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
degs = thetas_intra_df['deg']
band = thetas_intra_df['std'] * 2
mean = thetas_intra_df['mean']
ax = fig.add_subplot(1, 2, 1)
plt.errorbar(degs, mean, yerr=band)
