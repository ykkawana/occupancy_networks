# import torch.distributions as dist
import os
import pickle
os.environ['CUDA_PATH'] = '/usr/local/cuda-10.0'
import trimesh
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
from disposable import bsp_load_data
import random


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
if args.resume_generation_dir is not None:
    assert os.path.isabs(args.resume_generation_dir)

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

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if args.resume_generation_dir is None:
    out_dir = os.path.dirname(args.config)
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
else:
    out_dir = os.path.dirname(args.resume_generation_dir)
    generation_dir = args.resume_generation_dir

if not args.explicit:
    threshold_txt_path = os.path.join(out_dir, 'threshold')
    if os.path.exists(threshold_txt_path):
        with open(threshold_txt_path) as f:
            threshold = float(f.readlines()[0].strip())
            print('Use threshold in dir', threshold)
            cfg['test']['threshold'] = threshold

if not os.path.exists(generation_dir) and args.resume_generation_dir is None:
    os.makedirs(generation_dir)

if args.resume_generation_dir is None:
    patch_path = os.path.join(generation_dir, 'gen_diff.patch')
    subprocess.run('git diff > {}'.format(patch_path), shell=True)
    if not cfg['test']['model_file'].startswith('http'):
        weight_path = os.path.join(out_dir, cfg['test']['model_file'])
        with open(weight_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        cfg['test']['model_file_hash'] = md5
    yaml.dump(
        cfg,
        open(
            os.path.join(
                generation_dir,
                'gen_config_{}_{}.yaml'.format(args.unique_name, date_str)),
            'w'))

out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')
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

if 'modelid' in cfg['data']:
    points, values, pointcloud = bsp_load_data.get_bsp_data(
        cfg['data']['classes'][0] + '/' + cfg['data']['modelid'][0])
    input_idx = random.sample(range(len(pointcloud)),
                              k=min(cfg['data']['pointcloud_n'],
                                    len(pointcloud)))
    pointcloud_idx = random.sample(range(len(pointcloud)),
                                   k=min(cfg['data']['pointcloud_target_n'],
                                         len(pointcloud)))
    points_idx = random.sample(range(len(points)),
                               k=min(cfg['data']['points_subsample'],
                                     len(pointcloud)))
    batch_update = {
        'points':
        torch.from_numpy(points[points_idx, :]).unsqueeze(0),
        'points.occ':
        torch.from_numpy(values[points_idx, :]).unsqueeze(0).squeeze(-1),
        'pointcloud':
        torch.from_numpy(pointcloud[pointcloud_idx, :]).unsqueeze(0),
        'inputs':
        torch.from_numpy(pointcloud[input_idx, :]).unsqueeze(0)
    }
    val_batch_update = {
        'points_iou':
        torch.from_numpy(points[points_idx, :]).unsqueeze(0),
        'points_iou.occ':
        torch.from_numpy(values[points_idx, :]).unsqueeze(0).squeeze(-1),
        'pointcloud':
        torch.from_numpy(pointcloud[pointcloud_idx, :]).unsqueeze(0),
        'inputs':
        torch.from_numpy(pointcloud[input_idx, :]).unsqueeze(0)
    }
    batch = next(iter(test_loader))
    batch.update(batch_update)
    test_loader = [batch]

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
        is_mesh_file_exists = False
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        if os.path.exists(mesh_out_file):
            is_mesh_file_exists = True

        is_vertex_attribute_file_exists = False
        visibility_out_file = ''
        if cfg['generation'].get('is_explicit_mesh',
                                 False) and cfg['method'] == 'pnet':
            visibility_out_file = os.path.join(
                mesh_dir, '%s_vertex_visbility.npz' % modelname)
        elif cfg['method'] == 'bspnet':
            visibility_out_file = os.path.join(
                mesh_dir, '%s_%s.npz' % (modelname, cfg['test'].get(
                    'vertex_attribute_filename', 'vertex_attribute')))
        if os.path.exists(visibility_out_file):
            is_vertex_attribute_file_exists = True

        if is_input_file_exists and is_mesh_file_exists and is_vertex_attribute_file_exists and args.resume_generation_dir is not None:
            print('pass', category_id, modelname)
            continue

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

        # Write output
        if isinstance(mesh, list):
            #mesh.export(os.path.splitext(mesh_out_file)[0] + '.obj')
            mesh_out_file = os.path.splitext(mesh_out_file)[0] + '.pkl'
            pickle.dump(mesh, open(mesh_out_file, 'wb'))
            #mesh.export(os.path.splitext(mesh_out_file)[0] + '.obj')
            #print(os.path.splitext(mesh_out_file)[0] + '.obj')
        else:
            mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file
        if cfg['generation'].get('is_explicit_mesh',
                                 False) and cfg['method'] == 'pnet':
            visibility = mesh.vertex_attributes['vertex_visibility']

            np.savez(visibility_out_file, vertex_visibility=visibility)
            out_file_dict['vertex_visibility'] = visibility_out_file

        elif cfg['method'] == 'bspnet' and not cfg['generation'].get(
                'is_gen_skip_vertex_attributes', False):
            np.savez(visibility_out_file,
                     vertex_visibility=visibility,
                     vertices=vertices,
                     normals=normals)
            out_file_dict['vertex_attributes'] = visibility_out_file

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
            if cfg['method'] == 'bspnet':
                inputs = data['inputs'].squeeze(0).expand(3, -1, -1).cpu()
            else:
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
