# %%
# pylint: disable=not-callable
import torch
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
from collections import defaultdict
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
from im2mesh import eval
import eval_utils
from datetime import datetime
from tqdm import tqdm
import yaml
import dotenv
from bspnet import modelSVR
from bspnet import utils as bsp_utils

dotenv.load_dotenv('/home/mil/kawana/workspace/occupancy_networks/.env',
                   verbose=True)

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))


def separate_mesh_and_color(mesh_color_list):
    meshes = [mesh for mesh, _ in mesh_color_list]
    colors = [color for _, color in mesh_color_list]

    return meshes, colors


# %%
use_surface_only = True
debug = False

shapenetv1_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v1'
shapenetv2_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v2'
shapenetocc_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'

class_names = ['airplane', 'chair', 'lamp', 'table']

#base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/eval_config_20200502_105908.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/config.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/eval_config_20200502_172135.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_20200502_104902/eval_config_20200502_173040.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/eval_config_20200502_200733.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn30_20200501_191145/gen_part_label_20200504_210430.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn30_20200503_221032/eval_config_20200503_221032.yaml'
if len(sys.argv) > 1:
    config_path = sys.argv[1]
base_eval_dir = os.path.dirname(config_path)
semseg_shapenet_relative_path = 'data/ShapeNetBAESemSeg'

mesh_dir_path = os.path.join(base_eval_dir, 'generation_explicit')
fscore_pkl_path = os.path.join(mesh_dir_path,
                               'eval_fscore_from_meshes_full_explicit.pkl')

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

class_names_to_part_ids = {
    # id in partnet to id in here
    'airplane': {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    },
    'chair': {
        12: 0,
        13: 1,
        14: 2,
        15: 2
    },
    'table': {
        47: 0,
        48: 1,
        49: 1
    },
    'lamp': {
        24: 0,
        25: 1,
        26: 2,
        27: 2
    }
}

# %%
cfg = config.load_config(config_path, 'configs/default.yaml')
cfg['data']['classes'] = [label_to_synset[label] for label in class_names]

#cfg['model']['decoder_kwargs']['extract_surface_point_by_max'] = True
cfg['data']['semseg_path'] = semseg_shapenet_relative_path
cfg['test']['is_eval_semseg'] = True
cfg['data']['semseg_pointcloud_file'] = 'bae_semseg_labeled_pointcloud.npz'
cfg['data']['val_split'] = 'trainval'
cfg['data']['train_split'] = 'trainval'
cfg['data']['test_split'] = 'test'

if debug:
    #cfg['data']['debug'] = {'sample_n': 50}
    cfg['data']['classes'] = ['02691156']

bsp_use_cache = False
if 'part_label_out_dir' in cfg['generation']:
    bsp_use_cache = True

cfg['test']['is_eval_explicit_mesh'] = True
cfg['generation']['is_explicit_mesh'] = True
#eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

# Dataset
dataset = config.get_dataset('val', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(base_eval_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
val_loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         num_workers=0,
                                         shuffle=False)

# Generate
model.eval()

is_pnet = cfg['method'] == 'pnet'
is_atv2 = cfg['method'] == 'atlasnetv2'
is_bspnet = cfg['method'] == 'bspnet'

if is_pnet:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
elif is_atv2:
    nparts = cfg['model']['decoder_kwargs']['npatch']
elif is_bspnet:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
    gen_helper = modelSVR.BSPNetMeshGenerator(model, device=device)
else:
    raise NotImplementedError
label_counts = defaultdict(lambda: torch.zeros(nparts, 4).long())

# %%
for it, data in enumerate(tqdm(val_loader)):
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
    class_id = model_dict['category']
    class_name = synset_to_label[class_id]

    pointcloud = data.get('labeled_pointcloud').to(device)
    labels = data.get('labeled_pointcloud.labels').to(device)
    #normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
    inputs = data.get('inputs').to(device)

    kwargs = {}
    if is_pnet:
        point_scale = cfg['trainer']['pnet_point_scale']
    elif is_atv2:
        point_scale = cfg['trainer']['point_scale']
    elif is_bspnet:
        pass
    else:
        raise NotImplementedError

    #scaled_coord = points * point_scale
    if is_pnet:
        angles = data.get('angles').to(device)
        with torch.no_grad():
            feature = model.encode_inputs(inputs)
            output = model.decode(None, None, feature, angles=angles, **kwargs)
        vertices, vertices_mask, _, _, _ = output

        surface_vertices_mask = torch.where(
            vertices_mask.float() == 1, vertices_mask.float(),
            torch.tensor([np.inf], device=device).float())
        surface_vertices = vertices / point_scale * surface_vertices_mask.unsqueeze(
            -1)
    elif is_bspnet:
        if bsp_use_cache:
            vertex_attributes = np.load(
                os.path.join(
                    cfg['generation']['part_label_out_dir'], class_id,
                    '{}_primitive_vertex_visibility.npz'.format(modelname)))
            vertices = torch.from_numpy(
                vertex_attributes['points']).float().to(device).unsqueeze(0)
            vertices_mask = torch.from_numpy(
                vertex_attributes['vertex_visibility']).to(device).unsqueeze(0)
        else:
            with torch.no_grad():
                out_m, t = gen_helper.encode(inputs, measure_time=True)
                vertices, vertices_mask = gen_helper.sample_primitive_wise_surface_points(
                    out_m)
        if vertices is None or vertices_mask is None:
            print('vertices is None')
            continue

        surface_vertices_mask = torch.where(
            vertices_mask.float() == 1, vertices_mask.float(),
            torch.tensor([np.inf], device=device).float())
        B, N, P, dim = vertices.shape
        surface_vertices = (vertices *
                            surface_vertices_mask.unsqueeze(-1)).view(
                                B, N * P, dim)

        if not bsp_use_cache:
            surface_vertices = gen_helper.roty90(surface_vertices, inv=True)

            imnet_pointcloud = data.get('pointcloud.imnet_points').to(
                device).float()
            imnet_pointcloud = gen_helper.roty90(imnet_pointcloud, inv=True)

            surface_vertices = bsp_utils.realign(surface_vertices,
                                                 imnet_pointcloud, pointcloud)

        surface_vertices = surface_vertices.view(B, N, P, dim).contiguous()

    elif is_atv2:
        with torch.no_grad():
            debugged = cfg['training'].get('debugged', False)
            feature = model.encode_inputs(inputs *
                                          (1 if debugged else point_scale))
            patch = data.get('patch').to(device)
            surface_vertices = model.decode(
                None, None, feature, grid=patch, **kwargs) / point_scale

    B, N, P, dim = surface_vertices.shape
    dist, idx = eval.one_sided_chamfer_distance_with_index(
        pointcloud, surface_vertices.view(B, N * P, dim))
    assert not torch.any(torch.isinf(dist))

    sidx = (idx // P).view(-1).long()

    labels_long = labels.view(-1).long()
    for np_idx in range(nparts):
        for original_label, idx_idx in class_names_to_part_ids[
                class_name].items():
            label_counts[class_id][np_idx, idx_idx] += (
                (sidx == np_idx) &
                (labels_long == original_label)).long().sum()
counts_np = {}
for class_id, counts in label_counts.items():
    counts_np[class_id] = counts.to('cpu').numpy()
out_path = os.path.join(
    base_eval_dir, '{}part_assignment_{}.pkl'.format(
        ('debug_' if debug else ''), date_str))
pickle.dump(counts_np, open(out_path, 'wb'))
cfg['test']['semseg_part_assigned_pkl'] = out_path
yaml.dump(cfg, open(out_path.replace('.pkl', '.yaml'), 'w'))
print(counts_np)
# %%
