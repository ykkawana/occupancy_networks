import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict, OrderedDict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid
import numpy as np
import hashlib
import subprocess
import yaml
from datetime import datetime
import subprocess


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, represent_odict)
yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument(
    '--explicit',
    action='store_true',
    help=
    'to generate mesh with explicit rep, run: python3 generate.py --explicit --data.is_normal_icosahedron true --data.icosahedron_subdiv 4'
)
parser.add_argument('--unique_name',
                    default='',
                    type=str,
                    help='String name for generation.')

args, unknown_args = parser.parse_known_args()

cfg = config.load_config(args.config, 'configs/default.yaml')

for idx, arg in enumerate(unknown_args):
    if arg.startswith('--'):
        arg = arg.replace('--', '')
        value = unknown_args[idx + 1]
        keys = arg.split('.')
        if keys[0] not in cfg:
            cfg[keys[0]] = {}
        child_cfg = cfg.get(keys[0], {})
        for key in keys[1:]:
            item = child_cfg.get(key, None)
            if isinstance(item, dict):
                child_cfg = item
            elif item is None:
                if value == 'true':
                    value = True
                if value == 'false':
                    value = False
                if value == 'null':
                    value = None
                if isinstance(value, str) and value.isdigit():
                    value = float(value)
                child_cfg[key] = value
            else:
                child_cfg[key] = type(item)(value)
date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
if args.explicit:
    assert cfg['data'].get('is_normal_icosahedron', False) or cfg['data'].get(
        'is_normal_uv_sphere', False)
    cfg['generation']['is_explicit_mesh'] = True
    cfg['test']['is_eval_explicit_mesh'] = True
if args.explicit:
    cfg['generation']['generation_dir'] += '_explicit'
cfg['generation']['generation_dir'] += ('_' + date_str)

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = os.path.dirname(args.config)

generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

patch_path = os.path.join(generation_dir, 'gen_diff.patch')
subprocess.run('git diff > {}'.format(patch_path), shell=True)
weight_path = os.path.join(out_dir, cfg['test']['model_file'])
with open(weight_path, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
cfg['test']['model_file_hash'] = md5
yaml.dump(
    cfg,
    open(
        os.path.join(
            out_dir, 'gen_config_{}_{}.yaml'.format(args.unique_name,
                                                    date_str)), 'w'))

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')

# Loader
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          num_workers=0,
                                          shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(
        generation_dir,
        'vis',
    )

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    # Also copy ground truth
    if cfg['generation']['copy_groundtruth']:
        modelpath = os.path.join(dataset.dataset_folder, category_id,
                                 modelname, cfg['data']['watertight_file'])
        out_file_dict['gt'] = modelpath

    if generate_mesh:
        t0 = time.time()
        out = generator.generate_mesh(data)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        # Write output
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file
        if cfg['generation'].get('is_explicit_mesh', False):
            visibility = mesh.vertex_attributes['vertex_visibility']
            visibility_out_file = os.path.join(
                mesh_dir, '%s_vertex_visbility.npz' % modelname)
            np.savez(visibility_out_file, vertex_visibility=visibility)
            out_file_dict['vertex_visibility'] = visibility_out_file

    if generate_pointcloud:
        t0 = time.time()
        pointcloud = generator.generate_pointcloud(data)
        time_dict['pcl'] = time.time() - t0
        pointcloud_out_file = os.path.join(pointcloud_dir,
                                           '%s.ply' % modelname)
        export_pointcloud(pointcloud, pointcloud_out_file)
        out_file_dict['pointcloud'] = pointcloud_out_file

    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'img':
            inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            visualize_data(inputs, 'img', inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            voxel_mesh = VoxelGrid(inputs).to_mesh()
            voxel_mesh.export(inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir,
                                    '%02d_%s%s' % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
