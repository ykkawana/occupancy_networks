# %%
import trimesh
import os
import pandas
import pandas as pd
import pickle
import subprocess
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter
import sys
import tempfile
import open3d as o3d
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
sys.path.insert(
    0,
    '/home/mil/kawana/workspace/occupancy_networks/external/periodic_shapes')
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/atlasnetv2')
sys.path.insert(0, '/home/mil/kawana/workspace/superquadric_parsing')
from im2mesh.utils import binvox_rw
import math
import numpy as np
import matplotlib.pyplot as plt
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
import eval_utils
from tqdm import tqdm
import yaml
import pickle
from bspnet import utils as bsp_utils
from bspnet import modelSVR
from datetime import datetime
import random
import glob
import subprocess
from torch.utils.data import DataLoader
import torch_scatter
import pyquaternion
import scipy
random.seed(0)
project_dir = '/home/mil/kawana/workspace/occupancy_networks'
os.chdir(project_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/shapenet_svr_comparison'
date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')
out_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/resources/primitive_complexity'
rendering_out_dir = os.path.join(out_dir, 'rendering_{}'.format(date_str))
sq_mesh_out_dir = os.path.join(out_dir, 'sq_meshes')
sq_deciminated_mesh_out_dir = sq_mesh_out_dir + '_deciminated'
bsp_deciminated_mesh_out_dir = os.path.join(out_dir, 'bsp_meshes_deciminated')
SH_mesh_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes'
BSP_mesh_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/meshes'
SH_mesh_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes'
BSP_mesh_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/meshes'
class_names = ['airplane', 'chair']
sample_num = 500
# %%
SQ_commands = {
    'airplane':
    './save_mesh.py /data/unagi0/kawana/workspace/ShapeNetCore.v2/02691156/ {out_path} --model_tag "{model_id}" --n_primitives 20 --weight_file /home/mil/kawana/workspace/superquadric_parsing/scripts/wandb/run-20200201_111400-qe9b7jjp/model_149.pth  --train_with_bernoulli  --use_sq --dataset_type shapenet_v2 --save_prediction_as_mesh --use_deformation --run_on_gpu',
    'chair':
    './save_mesh.py /data/unagi0/kawana/workspace/ShapeNetCore.v2/03001627/ {out_path} --model_tag "{model_id}" --n_primitives 18 --weight_file /home/mil/kawana/workspace/superquadric_parsing/config/chair_T26AK2FES_model_699_py3 --train_with_bernoulli --use_deformations --use_sq --dataset_type shapenet_v2 --save_prediction_as_mesh --run_on_gpu'
}

# %%
mesh_lists = []
for class_name in class_names:
    class_id = label_to_synset[class_name]
    mesh_list = glob.glob(os.path.join(BSP_mesh_dir, class_id, '*.off'))
    mesh_list_selected = random.choices(mesh_list, k=sample_num)
    for mesh_path in mesh_list_selected:
        model_id = os.path.splitext(os.path.basename(mesh_path))[0]
        sh_mesh_path = os.path.join(SH_mesh_dir, class_id, model_id + '.off')
        assert os.path.exists(sh_mesh_path)
        mesh_lists.append((class_id, model_id))

# %%
mesh_list_per_class = defaultdict(lambda: [])
for class_id, model_id in mesh_lists:
    mesh_list_per_class[class_id].append(model_id)

for class_name in class_names:
    class_id = label_to_synset[class_name]
    if not os.path.exists(os.path.join(out_dir, class_id + '.txt')):
        with open(os.path.join(out_dir, class_id + '.txt'), 'w') as f:
            f.write('\n'.join(mesh_list_per_class[class_id]))

# %%
sh_mesh_paths = [
    os.path.join(SH_mesh_dir, class_id, model_id + '.off')
    for class_id, model_id in mesh_lists
]
bsp_mesh_paths = [
    os.path.join(BSP_mesh_dir, class_id, model_id + '.off')
    for class_id, model_id in mesh_lists
]
sq_mesh_paths = glob.glob(os.path.join(
    sq_mesh_out_dir, '02691156', '*.off')) + glob.glob(
        os.path.join(sq_mesh_out_dir, '03001627', '*.off'))

mesh_paths = OrderedDict({
    'sh': sh_mesh_paths,
    'bsp': bsp_mesh_paths,
    'sq': sq_mesh_paths
})

random_indices = random.choices(range(700), k=10)
# %%
items = []
pbar = tqdm(total=sum([len(paths) for paths in mesh_paths.values()]))
for model_name, paths in mesh_paths.items():
    print(model_name)
    cnt = 0
    rendered = 0
    for path in paths:
        pbar.update(1)
        if rendered > 9:
            continue
        mesh = trimesh.load(path)
        meshes = mesh.split()
        if meshes is None or len(meshes) == 0:
            meshes = [mesh]
        for mesh_i in meshes:
            cnt += 1
            if cnt in random_indices:
                rendered += 1
                mesh_i.vertices = eval_utils.normalize_verts_in_occ_way(
                    mesh_i.vertices)
                with tempfile.NamedTemporaryFile(suffix='.off') as f:
                    mesh_i.export(f.name)
                    eval_utils.render_by_blender(rendering_script_path,
                                                 camera_param_path,
                                                 f.name,
                                                 rendering_out_dir,
                                                 '{}_{}'.format(
                                                     model_name, cnt),
                                                 use_cycles=True,
                                                 use_lamp=True)

# %%
