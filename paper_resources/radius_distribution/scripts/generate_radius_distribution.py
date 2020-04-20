# %%
import argparse
import os
import subprocess
import hashlib
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks/im2mesh')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import shutil
import yaml
from collections import OrderedDict

# %%
config_path = ''
resource_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/radius_distribution/resources'
use_cache_flag = False

# %%


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

# %%
# Get configuration and basic arguments
cfg = config.load_config(config_path, 'configs/default.yaml')
cfg['data']['icosahedron_uv_margin'] = 0
cfg['data']['icosahedron_uv_margin_phi'] = 0
cfg['data']['points_subsample'] = 2
cfg['data']['debug'] = {'sample_n': 50}

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
thetas = np.linspace(-np.pi, np.pi - 2 * np.pi / 100, 100)
phis = np.linspace(-np.pi / 2, np.pi / 2 - np.pi / 100, 100)
theta_radius_list = []
phi_radius_list = []
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
    inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
    theta = torch.meshgrid(
        [torch.tensor(thetas, device=device),
         torch.zeros([1], device=device)],
        device=device).unsqueeze(0)
    phi = torch.meshgrid(
        [torch.zeros([1], device=device),
         torch.tensor(phis, device=device)],
        device=device).unsqueeze(0)
    points = data.get('points').to(device)

    feature = model.encode_inputs(inputs)

    kwargs = {}
    scaled_coord = points * cfg['trainer']['pnet_point_scale']
    output_theta = model.decode(scaled_coord,
                                None,
                                feature,
                                angles=theta,
                                **kwargs)
    _, _, _, _, theta_radius = output_theta
    theta_radius_list.append(theta_radius[:, :, :,
                                          0].mean(axis=(0, 1)).cpu().numpy())
    output_phi = model.decode(scaled_coord,
                              None,
                              feature,
                              angles=phi,
                              **kwargs)
    _, _, _, _, phi_radius = output_phi
    phi_radius_list.append(phi_radius[:, :, :,
                                      1].mean(axis=(0, 1)).cpu().numpy())

# Create pandas dataframe and save
theta_df = pd.DataFrame(theta_radius_list, columns=thetas).mean()
theta_df.to_pickle('test')

print(theta_df)
