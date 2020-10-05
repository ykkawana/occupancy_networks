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
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh import eval
import eval_utils
from datetime import datetime
from tqdm import tqdm
import yaml
import dotenv
from bspnet import modelSVR
from bspnet import utils as bsp_utils
import warnings
from joblib import Parallel, delayed
import pandas as pd
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

config_path = sys.argv[1]
mesh_dir_path = os.path.dirname(config_path)
out_path = os.path.join(mesh_dir_path,
                        'surface_verts_faces_count_sample100k.pkl')
base_eval_dir = os.path.dirname(mesh_dir_path)
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
class_names = list(synset_to_label.values())
cfg = config.load_config(config_path, 'configs/default.yaml')
cfg['data']['classes'] = [label_to_synset[label] for label in class_names]
is_explicit_mesh = cfg['generation'].get('is_explicit_mesh', False)
is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

is_pnet = cfg['method'] == 'pnet'
is_atv2 = cfg['method'] == 'atlasnetv2'
is_bspnet = cfg['method'] == 'bspnet'

# %%
# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(base_eval_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False)

model.eval()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


num_proc = 30
indices_list = np.array_split(np.arange(len(dataset)), num_proc)


def get_verts_faces_stats(inputs):
    print('start parallel')
    indices, dataset = inputs

    return_dicts = []
    for it in tqdm(indices):
        #if it > 10:
        #    break

        data = dataset[it]
        if data is None:
            print('Invalid data.')
            continue
        # Get index etc.
        idx = data['idx']

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        class_id = model_dict['category']
        if class_id == 'n/a':
            class_name = 'n/a'
            continue
        else:
            class_name = synset_to_label[class_id]

        kwargs = {}

        visibility_path = os.path.join(
            mesh_dir_path, 'meshes', class_id,
            modelname + '_vertex_attributes_sample100k.npz')
        mesh_path = os.path.join(mesh_dir_path, 'meshes', class_id,
                                 modelname + '.off')
        try:
            all_mesh = trimesh.load(mesh_path)
        except:
            print('error', mesh_path, visibility_path)
            continue
        verts, idx = all_mesh.sample(100000, return_index=True)
        normals = all_mesh.face_normals[idx, :]
        surface_verts = verts + normals * 1e-4

        meshes = all_mesh.split()
        for idx, mesh in enumerate(meshes):
            if idx == 0:
                occ = check_mesh_contains(mesh, surface_verts)
            else:
                occ |= check_mesh_contains(mesh, surface_verts)

        v1 = all_mesh.faces[:, 0]
        v2 = all_mesh.faces[:, 1]
        v3 = all_mesh.faces[:, 2]

        v1_in = np.take(occ, v1)
        v2_in = np.take(occ, v2)
        v3_in = np.take(occ, v3)
        face_in = v1_in & v2_in & v3_in
        return_dicts.append({
            'class_id': class_id,
            'class_name': class_name,
            'modelname': modelname,
            'verts_num': len(occ),
            'faces_num': len(face_in),
            'verts_in': occ.sum(),
            'faces_in': face_in.sum(),
            'verts_out': len(occ) - occ.sum(),
            'faces_out': len(face_in) - face_in.sum()
        })
        np.savez(visibility_path,
                 normals=normals,
                 vertices=verts,
                 vertex_visibility=np.logical_not(occ))

    return return_dicts


# %%
r = Parallel(n_jobs=num_proc)([
    delayed(get_verts_faces_stats)((indices, dataset))
    for indices in indices_list
])

all_dicts = []
for dicts in r:
    all_dicts.extend(dicts)

df = pd.DataFrame(all_dicts)
df.to_pickle(out_path)
