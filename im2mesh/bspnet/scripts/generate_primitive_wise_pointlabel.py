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
import warnings
warnings.simplefilter('ignore')

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
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/eval_config_20200502_200733.yaml'
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
assert cfg['method'] == 'bspnet'
#cfg['model']['decoder_kwargs']['extract_surface_point_by_max'] = True
cfg['data']['semseg_path'] = semseg_shapenet_relative_path
cfg['test']['is_eval_semseg'] = True
cfg['data']['semseg_pointcloud_file'] = 'bae_semseg_labeled_pointcloud.npz'
cfg['data']['val_split'] = 'trainval'
cfg['data']['train_split'] = 'trainval'
cfg['data']['test_split'] = 'test'
cfg['generation']['part_label_out_dir'] = os.path.join(
    base_eval_dir, 'part_label_generation_{}'.format(date_str))

cfg['test']['is_eval_explicit_mesh'] = True
cfg['generation']['is_explicit_mesh'] = True
#eval_utils.update_dict_with_options(cfg, unknown_args)
if not debug:
    os.makedirs(cfg['generation']['part_label_out_dir'])
    yaml.dump(
        cfg,
        open(
            os.path.join(base_eval_dir,
                         'gen_part_label_{}.config').format(date_str), 'w'))

is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

is_pnet = cfg['method'] == 'pnet'
is_atv2 = cfg['method'] == 'atlasnetv2'
is_bspnet = cfg['method'] == 'bspnet'

nparts = cfg['model']['decoder_kwargs']['n_primitives']

# %%
splits = ['test', 'val']
for split in splits:
    # Dataset
    dataset = config.get_dataset(split, cfg, return_idx=True)

    # Model
    model = config.get_model(cfg, device=device, dataset=dataset)

    checkpoint_io = CheckpointIO(base_eval_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # Loader
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         num_workers=0,
                                         shuffle=False)

    gen_helper = modelSVR.BSPNetMeshGenerator(model, device=device)
    # Generate
    model.eval()

    for it, data in enumerate(tqdm(loader)):
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
        #scaled_coord = points * point_scale
        if is_pnet:
            raise ValueError
        elif is_bspnet:
            with torch.no_grad():
                try:
                    out_m, t = gen_helper.encode(inputs, measure_time=True)
                    vertices, vertices_mask = gen_helper.sample_primitive_wise_surface_points(
                        out_m)
                except:
                    continue
            if vertices is None or vertices_mask is None:
                print('vertices is None')
                continue

            B, N, P, dim = vertices.shape
            surface_vertices = vertices

            surface_vertices = gen_helper.roty90(surface_vertices.view(
                B, N * P, dim),
                                                 inv=True)

            imnet_pointcloud = data.get('pointcloud.imnet_points').to(
                device).float()
            imnet_pointcloud = gen_helper.roty90(imnet_pointcloud, inv=True)

            surface_vertices = bsp_utils.realign(
                surface_vertices, imnet_pointcloud,
                pointcloud).view(B, N, P, dim).contiguous()
            vertices_mask = vertices_mask.view(B, N, P)
            out_dir = cfg['generation']['part_label_out_dir']
            if not debug:
                class_out_dir = os.path.join(out_dir, class_id)
                if not os.path.exists(class_out_dir):
                    os.makedirs(class_out_dir)
                out_path = os.path.join(
                    out_dir, class_id,
                    '{}_primitive_vertex_visibility.npz'.format(modelname))
                np.savez(out_path,
                         points=surface_vertices[0].cpu().numpy(),
                         vertex_visibility=vertices_mask[0].cpu().numpy())
