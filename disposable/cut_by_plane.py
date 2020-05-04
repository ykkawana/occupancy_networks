
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
from im2mesh import data
import skimage.io

dotenv.load_dotenv('/home/mil/kawana/workspace/occupancy_networks/.env',
                   verbose=True)

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))


def separate_mesh_and_color(mesh_color_list):
    meshes = [mesh for mesh, _ in mesh_color_list]
    colors = [color for _, color in mesh_color_list]

    return meshes, colors


# %%
use_surface_only = True
debug = True
shapenetv1_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v1'
shapenetv2_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v2'
shapenetocc_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'

class_names = ['airplane', 'chair', 'lamp', 'table']

#base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/eval_config_20200502_105908.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/eval_config_20200502_200733.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/config.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn30_20200501_191145/config.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/eval_config_20200502_172135.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_20200502_104902/eval_config_20200502_173040.yaml'
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

cfg['test']['is_eval_explicit_mesh'] = True
cfg['generation']['is_explicit_mesh'] = True
#eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")


# Dataset
dataset = config.get_dataset('val', cfg, return_idx=True)
dataset.fields['patch'] = data.PlanarPatchField(
        'test', cfg['data'].get('patch_side_length', 200),
        True)


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

if is_pnet:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
else:
    raise NotImplementedError

# %%
data = dataset[10]
if data is None:
    print('Invalid data.')
# Get index etc.
idx = data['idx']

try:
    model_dict = dataset.get_model_dict(idx)
except AttributeError:
    model_dict = {'model': str(idx), 'category': 'n/a'}

modelname = model_dict['model']
class_id = model_dict['category']
class_name = synset_to_label[class_id]

pointcloud = torch.from_numpy(data.get('labeled_pointcloud')).float().to(device).unsqueeze(0)
#normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
inputs = data.get('inputs').to(device).unsqueeze(0)

kwargs = {}
if is_pnet:
    point_scale = cfg['trainer']['pnet_point_scale']
else:
    raise NotImplementedError

#scaled_coord = points * point_scale
angles = data.get('angles').to(device).unsqueeze(0)
with torch.no_grad():
    coord = (data.get('patch.mesh_vertices').unsqueeze(0).float() - 0.5) * point_scale
    coord = torch.cat([coord, torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1)], axis=-1).to(device)
    #coord = torch.cat([coord[:, :, 0].unsqueeze(-1), torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1), coord[:, :, 1].unsqueeze(-1) ], axis=-1).to(device)

    output = model(coord, inputs, sample=True, angles=angles, **kwargs)
vertices, vertices_mask, sgd, _, _ = output

print(class_id, modelname)
si = sgd.max(1)[0].view(200, 200)

wb = si > 0

skimage.io.imsave('test.png', wb.cpu().long().numpy().astype(np.uint8)*255)
print(inputs.shape)
skimage.io.imsave('inputs.png', 255 *inputs[0, :, :].mean(0).cpu().numpy().astype(np.uint8))
