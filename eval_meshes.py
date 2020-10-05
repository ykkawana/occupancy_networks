import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from torch.utils import data as torch_data
from im2mesh import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud
import numpy as np
from datetime import datetime
import yaml
import eval_utils

parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input',
                    action='store_true',
                    help='Evaluate inputs instead.')

parser.add_argument('--unique_name',
                    default='',
                    type=str,
                    help='String name for generation.')
args, unknown_args = parser.parse_known_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

is_eval_explicit_mesh = cfg['test'].get('is_eval_explicit_mesh', False)
# Shorthands

out_dir = os.path.dirname(args.config)
generation_dir = os.path.dirname(args.config)
date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
assert generation_dir.endswith(cfg['generation']['generation_dir'])
if not args.eval_input:
    out_file = os.path.join(
        generation_dir, 'eval_meshes_full_{}_{}_{}.pkl'.format(
            args.unique_name, '_explicit' if is_eval_explicit_mesh else '',
            date_str))
    out_file_class = os.path.join(
        generation_dir, 'eval_meshes_{}_{}_{}.csv'.format(
            args.unique_name, '_explicit' if is_eval_explicit_mesh else '',
            date_str))
else:
    out_file = os.path.join(
        generation_dir, 'eval_input_full_{}_{}_{}.pkl'.format(
            args.unique_name, '_explicit' if is_eval_explicit_mesh else '',
            date_str))
    out_file_class = os.path.join(
        generation_dir, 'eval_input_{}_{}_{}.csv'.format(
            args.unique_name, '_explicit' if is_eval_explicit_mesh else '',
            date_str))

# Dataset
points_field = data.PointsField(
    cfg['data']['points_iou_file'],
    unpackbits=cfg['data']['points_unpackbits'],
)
pointcloud_field = data.PointCloudField(cfg['data']['pointcloud_chamfer_file'])
fields = {
    'points_iou': points_field,
    'pointcloud_chamfer': pointcloud_field,
    'idx': data.IndexField(),
}

print('Test split: ', cfg['data']['test_split'])

dataset_folder = cfg['data']['path']
dataset = data.Shapes3dDataset(dataset_folder,
                               fields,
                               cfg['data']['test_split'],
                               categories=cfg['data']['classes'])

if 'debug' in cfg['data']:
    dataset = torch_data.Subset(dataset,
                                range(cfg['data']['debug']['sample_n']))
yaml.dump(
    cfg,
    open(
        os.path.join(
            generation_dir,
            'eval_mesh_config_{}_{}.yaml'.format(args.unique_name, date_str)),
        'w'))

# Evaluator
evaluator = MeshEvaluator(
    n_points=cfg['test']['n_points'],
    is_sample_from_surface=cfg['test']['is_sample_from_surface'],
    is_normalize_by_side_length=cfg['test']['is_normalize_by_side_length'],
    is_eval_iou_by_split=cfg['test'].get('is_eval_iou_by_split', False))

# Loader
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          num_workers=0,
                                          shuffle=False)
vertex_attribute_filename = 'vertex_attributes'
if cfg['test'].get('eval_from_vertex_attributes', False):
    vertex_attribute_filename = cfg['test']['vertex_attribute_filename']
    cfg['method'] = 'bspnet'
if cfg['method'] == 'bspnet' and cfg['generation'].get('is_gen_implicit_mesh',
                                                       False):
    is_eval_explicit_mesh = False
    cfg['method'] = 'onet'
# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

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

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
    normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
    points_tgt = data['points_iou'].squeeze(0).numpy()
    occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_for_iou = None
        vertex_visibility = None
        if cfg['method'] == 'pnet':
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file, process=False)
            else:
                print('Warning: mesh file does not exist: %s' % mesh_file)
                continue
            if is_eval_explicit_mesh:
                visbility_file = os.path.join(
                    mesh_dir, '%s_vertex_visbility.npz' % modelname)
                if os.path.exists(visbility_file):
                    vertex_visibility = np.load(
                        visbility_file)['vertex_visibility']
                else:
                    print('Warning: vibility file does not exist: %s' %
                          visbility_file)
                    continue
        elif cfg['method'] == 'bspnet':
            vertex_file = os.path.join(
                mesh_dir, '%s_%s.npz' % (modelname, vertex_attribute_filename))
            is_eval_explicit_mesh = True
            if os.path.exists(vertex_file):
                try:
                    vertex_attributes = np.load(vertex_file)
                    verts = vertex_attributes['vertices']
                    trargs = [verts]
                    try:
                        normals = vertex_attributes['normals']
                        trargs.append(normals)
                    except:
                        pass
                    mesh = trimesh.Trimesh(*trargs)
                    mesh_for_iou = trimesh.load(
                        os.path.join(mesh_dir, '{}.off'.format(modelname)))
                    assert isinstance(mesh_for_iou, trimesh.Trimesh)
                except:
                    print('Error in bspnet loading vertex')
                    continue
                try:
                    vertex_visibility = vertex_attributes['vertex_visibility']
                except:
                    vertex_visibility = None
            else:
                print('Warning: vertex file does not exist: %s' % vertex_file)
                continue
        elif cfg['method'] == 'atlasnetv2':
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)
            is_eval_explicit_mesh = False

            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file, process=False)
            else:
                print('Warning: mesh file does not exist: %s' % mesh_file)
                continue
        else:
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file, process=False)
            else:
                print('Warning: mesh file does not exist: %s' % mesh_file)
                continue

        eval_dict_mesh = evaluator.eval_mesh(
            mesh,
            pointcloud_tgt,
            normals_tgt,
            points_tgt,
            occ_tgt,
            mesh_for_iou=mesh_for_iou,
            is_eval_explicit_mesh=is_eval_explicit_mesh,
            vertex_visibility=vertex_visibility,
            skip_iou=(cfg['method'] == 'atlasnetv2'))
        for k, v in eval_dict_mesh.items():
            eval_dict[k + ' (mesh)'] = v

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(pointcloud_dir, '%s.ply' % modelname)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(pointcloud,
                                                      pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s' % pointcloud_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
