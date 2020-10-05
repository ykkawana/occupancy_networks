# import torch.distributions as dist
import os
os.environ['CUDA_PATH'] = '/usr/local/cuda-10.0'

import torch
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
import eval_utils


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
parser.add_argument('--resume_generation_dir',
                    default=None,
                    type=str,
                    help='String name for generation.')

args, unknown_args = parser.parse_known_args()

cfg = config.load_config(args.config, 'configs/default.yaml')

eval_utils.update_dict_with_options(cfg, unknown_args)

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
if args.explicit:
    if cfg['method'] == 'pnet':
        assert cfg['data'].get('is_normal_icosahedron',
                               False) or cfg['data'].get(
                                   'is_normal_uv_sphere', False)
    elif cfg['method'] == 'atlasnetv2':
        assert cfg['data'].get('is_generate_mesh', False)

    cfg['generation']['is_explicit_mesh'] = True
    cfg['test']['is_eval_explicit_mesh'] = True
    cfg['generation']['generation_dir'] += '_explicit'

cfg['generation']['generation_dir'] += ('_' + args.unique_name + '_' +
                                        date_str)
if cfg['method'] == 'bspnet':
    cfg['generation']['is_gen_primitive_wise_watertight_mesh'] = True
    cfg['generation']['is_gen_skip_vertex_attributes'] = True

elif cfg['method'] == 'pnet':
    cfg['generation']['is_skip_surface_mask_generation_time'] = True
generation_dir = os.path.dirname(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = os.path.dirname(generation_dir)
if not args.explicit:
    threshold_txt_path = os.path.join(out_dir, 'threshold')
    if os.path.exists(threshold_txt_path):
        with open(threshold_txt_path) as f:
            threshold = float(f.readlines()[0].strip())
            print('Use threshold in dir', threshold)
            cfg['test']['threshold'] = threshold

out_time_file = os.path.join(generation_dir,
                             'eval_gen_time_full_{}.pkl'.format(date_str))
out_time_file_class = os.path.join(generation_dir,
                                   'eval_gen_time_{}.pkl'.format(date_str))
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
checkpoint_io.load(cfg['test']['model_file'], device=device)

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
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


np.random.seed(0)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          num_workers=0,
                                          worker_init_fn=worker_init_fn,
                                          shuffle=True)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

measure_num = 500

for it, data in enumerate(tqdm(test_loader)):
    if it > measure_num:
        break
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
        # Checkfile exists
        is_input_file_exists = False
        if input_type == 'img':
            inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
        elif input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
        elif input_type == 'pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
        if os.path.exists(input_type):
            is_input_file_exists = True

        # Write output

        t0 = time.time()
        out = generator.generate_mesh(data)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        if cfg['method'] == 'bspnet':
            try:
                mesh, stats_dict, vertices, normals, visibility = out
            except TypeError:
                mesh, vertices, normals, visibility = out
                stats_dict = {}
            if mesh is None:
                continue
        else:
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

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
