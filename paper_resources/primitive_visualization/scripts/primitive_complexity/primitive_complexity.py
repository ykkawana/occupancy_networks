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
out_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/resources/primitive_complexity'
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

# %%
# Get nsd pirimitive mesh
nsd_mesh = trimesh.creation.icosphere(subdivisions=4)
nsd_verts_len = len(nsd_mesh.vertices)
nsd_faces_len = len(nsd_mesh.faces)
# %%
# Deciminate sq mesh.
pbar = tqdm(total=len(sq_mesh_paths))
for path in sq_mesh_paths:
    print(path)
    pbar.update(1)
    deci_path = path.replace('sq_meshes', 'sq_meshes_deciminated')
    if os.path.exists(deci_path):
        continue
    if not os.path.exists(os.path.dirname(deci_path)):
        os.makedirs(os.path.dirname(deci_path))
    mesh = trimesh.load(path)
    meshes = mesh.split()
    if meshes is None or len(meshes) == 0:
        meshes = [mesh]
    deci_meshes = []
    for mesh_i in meshes:
        om_mesh_i = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_i.vertices),
            o3d.utility.Vector3iVector(mesh_i.faces))
        deci_om_mesh_i = o3d.geometry.TriangleMesh.simplify_quadric_decimation(
            om_mesh_i, nsd_faces_len)
        deci_mesh_i = trimesh.Trimesh(np.asarray(deci_om_mesh_i.vertices),
                                      np.asarray(deci_om_mesh_i.triangles))
        deci_meshes.append(deci_mesh_i)
    trimesh.util.concatenate(deci_meshes).export(deci_path)
# %%
# Subdivide and Deciminate bsp mesh.
pbar = tqdm(total=len(bsp_mesh_paths))
for path in bsp_mesh_paths:
    pbar.update(1)
    deci_path = os.path.join(bsp_deciminated_mesh_out_dir,
                             *path.split('/')[-2:])

    if os.path.exists(deci_path):
        continue
    if not os.path.exists(os.path.dirname(deci_path)):
        os.makedirs(os.path.dirname(deci_path))

    mesh = trimesh.load(path)
    meshes = mesh.split()
    if meshes is None or len(meshes) == 0:
        meshes = [mesh]
    deci_meshes = []
    for mesh_i in meshes:
        mesh_subdiv = mesh_i.copy()
        trimesh.repair.fix_inversion(mesh_subdiv)
        trimesh.repair.fix_winding(mesh_subdiv)
        trimesh.repair.fix_normals(mesh_subdiv)
        while 2300 > len(mesh_subdiv.faces):
            mesh_subdiv = trimesh.Trimesh(*trimesh.remesh.subdivide(
                mesh_subdiv.vertices, mesh_subdiv.faces))
        """
        om_mesh_subdiv = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_subdiv.vertices),
            o3d.utility.Vector3iVector(
                mesh_subdiv.faces)).compute_convex_hull()[0]

        while len(np.asarray(om_mesh_subdiv.triangles)) < nsd_faces_len:
            om_mesh_subdiv = o3d.geometry.TriangleMesh.subdivide_loop(
                om_mesh_subdiv)
        om_mesh_i = om_mesh_subdiv
        """
        """
        om_mesh_i = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_subdiv.vertices),
            o3d.utility.Vector3iVector(mesh_subdiv.faces))
        deci_om_mesh_i = o3d.geometry.TriangleMesh.simplify_quadric_decimation(
            om_mesh_i, nsd_faces_len)
        deci_mesh_i = trimesh.Trimesh(np.asarray(deci_om_mesh_i.vertices),
                                      np.asarray(deci_om_mesh_i.triangles))
        """
        deci_mesh_i = mesh_subdiv
        deci_meshes.append(deci_mesh_i)
    trimesh.util.concatenate(deci_meshes).export(deci_path)
# %%
# Sphericity

items = []
if not os.path.exists(os.path.join(out_dir, 'sphericity.pkl')):
    pbar = tqdm(total=sum([len(paths) for paths in mesh_paths.values()]))
    for model_name, paths in mesh_paths.items():
        for path in paths:
            pbar.update(1)
            mesh = trimesh.load(path)
            meshes = mesh.split()
            if meshes is None or len(meshes) == 0:
                meshes = [mesh]
            for mesh_i in meshes:
                mesh_i.vertices = eval_utils.normalize_verts_in_occ_way(
                    mesh_i.vertices)
                axis = mesh_i.vertices.max(0) - mesh_i.vertices.min(0)
                axis_norm = axis / (np.linalg.norm(axis))
                xaxis = np.array([0., 1., 0.])
                angles = np.arccos(np.dot(axis_norm, xaxis))
                q = pyquaternion.Quaternion(axis=(xaxis - axis_norm),
                                            angle=-angles)
                r = scipy.spatial.transform.Rotation(q.elements)
                mesh_i.vertices = r.apply(mesh_i.vertices)
                mesh_i.vertices = (mesh_i.vertices - mesh_i.vertices.min(0)
                                   ) / (mesh_i.vertices.max(0) -
                                        mesh_i.vertices.min(0))
                area = mesh_i.area
                volume = mesh_i.volume
                sphericity = ((36 * np.pi * volume**2)**(1 / 3)) / (area)
                items.append({
                    'model_name': model_name,
                    'area': area,
                    'volume': volume,
                    'sphericity': sphericity
                })
    df = pd.DataFrame(items)
    df.to_pickle(os.path.join(out_dir, 'sphericity.pkl'))
# %%

items = []
if not os.path.exists(os.path.join(out_dir, 'gaussian_curvature.pkl')) or True:
    pbar = tqdm(total=sum([len(paths) for paths in mesh_paths.values()]))
    for model_name, paths in mesh_paths.items():
        for path in paths:
            """
            if model_name == 'bsp':
                path = os.path.join(bsp_deciminated_mesh_out_dir,
                                    *path.split('/')[-2:])
            elif model_name == 'sq':
                path = path.replace('sq_meshes', 'sq_meshes_deciminated')
            """
            pbar.update(1)
            mesh = trimesh.load(path)
            meshes = mesh.split()
            if meshes is None or len(meshes) == 0:
                meshes = [mesh]
            for mesh_i in meshes:
                trimesh.repair.fix_inversion(mesh_i)
                trimesh.repair.fix_winding(mesh_i)
                trimesh.repair.fix_normals(mesh_i)
                mesh_i.vertices = eval_utils.normalize_verts_in_occ_way(
                    mesh_i.vertices)
                area_per_face = mesh_i.area_faces
                area_per_face = np.concatenate([area_per_face, np.array([0])])
                vertex_faces = mesh_i.vertex_faces
                area_per_vertex = np.abs(
                    np.take(area_per_face, vertex_faces, axis=0)).sum(1)
                face_angles = mesh_i.face_angles
                vertex_angles_sum = torch_scatter.scatter_add(
                    torch.from_numpy(face_angles).view(-1),
                    torch.from_numpy(mesh_i.faces).view(-1)).numpy()
                gauss = np.abs(
                    (2 * np.pi - vertex_angles_sum) / (area_per_vertex / 3))
                trimesh_gauss_curv = trimesh.curvature.discrete_gaussian_curvature_measure(
                    mesh_i, mesh_i.vertices, 0.1)
                trimesh_mean_curv = trimesh.curvature.discrete_mean_curvature_measure(
                    mesh_i, mesh_i.vertices, 0.1)
                items.append({
                    'model_name':
                    model_name,
                    'faces':
                    len(mesh_i.faces),
                    'vertices':
                    len(mesh_i.vertices),
                    'gaussian_curvature_mean':
                    gauss.mean(),
                    'gaussian_curvature_std':
                    gauss.std(),
                    'area_mean':
                    area_per_vertex.mean(),
                    'area_std':
                    area_per_vertex.std(),
                    'vertex_angles_mean':
                    face_angles.mean(),
                    'vertex_angles_std':
                    face_angles.std(),
                    'pos_trimesh_gauss_curv_mean':
                    trimesh_gauss_curv[trimesh_gauss_curv >= 0].mean(),
                    'pos_trimesh_gauss_curv_std':
                    trimesh_gauss_curv[trimesh_gauss_curv >= 0].std(),
                    'pos_trimesh_mean_curv_mean':
                    trimesh_mean_curv[trimesh_mean_curv >= 0].mean(),
                    'pos_trimesh_mean_curv_std':
                    trimesh_mean_curv[trimesh_mean_curv >= 0].std(),
                    'neg_trimesh_gauss_curv_mean':
                    trimesh_gauss_curv[trimesh_gauss_curv <= 0].mean(),
                    'neg_trimesh_gauss_curv_std':
                    trimesh_gauss_curv[trimesh_gauss_curv <= 0].std(),
                    'neg_trimesh_mean_curv_mean':
                    trimesh_mean_curv[trimesh_mean_curv <= 0].mean(),
                    'neg_trimesh_mean_curv_std':
                    trimesh_mean_curv[trimesh_mean_curv <= 0].std(),
                    'abs_trimesh_gauss_curv_mean':
                    np.abs(trimesh_gauss_curv).mean(),
                    'abs_trimesh_gauss_curv_std':
                    np.abs(trimesh_gauss_curv).std(),
                    'abs_trimesh_mean_curv_mean':
                    np.abs(trimesh_mean_curv).mean(),
                    'abs_trimesh_mean_curv_std':
                    np.abs(trimesh_mean_curv).std(),
                })
    df = pd.DataFrame(items)
    df.to_pickle(  #os.path.join(out_dir, 'gaussian_curvature_even_faces_01.pkl'))
        os.path.join(out_dir, 'gaussian_curvature_uneven_faces_01.pkl'))
    print(df.groupby('model_name').mean())
# %%
