# %%
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
import open3d as o3d
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
import eval_utils
from tqdm import tqdm
import yaml
import pickle
from bspnet import utils as bsp_utils
from bspnet import modelSVR
from datetime import datetime
import eval_utils
import pypoisson
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# %%
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/gen_config_f60k_20200511_163104.yaml'
base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018'
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
cfg = config.load_config(config_path, 'configs/default.yaml')
#cfg['data']['is_normal_icosahedron'] = True
#cfg['data']['icosahedron_subdiv'] = 4
assert cfg['method'] == 'pnet'
#eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(base_eval_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          num_workers=4,
                                          shuffle=False)

# Generate
model.eval()

# %%
indices = {}
for idx in range(len(dataset)):
    model_dict = dataset.get_model_dict(idx)
    model_id = model_dict['model']
    class_id = model_dict.get('category', 'n/a')
    indices[(class_id, model_id)] = idx
# %%
target_class = 'car'
target_model_id = 'cbe2dc469c47bb80425b2c354eccabaf'
target_class = 'chair'
target_model_id = 'cbc9014bb6ce3d902ff834514c92e8fd'
idx = indices[(label_to_synset[target_class], target_model_id)]
data = dataset[idx]

normal_angles = data.get('angles.normal_angles').to(device).unsqueeze(0)
normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
inputs = data.get('inputs').to(device).unsqueeze(0)
feature = model.encode_inputs(inputs)

kwargs = {}
point_scale = cfg['trainer']['pnet_point_scale']

#scaled_coord = points * point_scale
model.decoder.p_sampler.extract_surface_point_by_max = False
with torch.no_grad():
    output = model.decode(None, None, feature, angles=normal_angles, **kwargs)
    normal_vertices, mask, _, _, _ = output

mask = mask.view(-1).to('cpu').detach().numpy()
verts = (normal_vertices[0, ...] / point_scale).to('cpu').detach().numpy()
npart, verts_n, _ = verts.shape
verts_all = verts.reshape([-1, 3])
faces = normal_faces[0, ...].to('cpu').detach().numpy()
faces_all = np.concatenate([faces + verts_n * i for i in range(npart)])

# For primitive wise rendering
mesh = trimesh.Trimesh(verts_all, faces_all, process=False)
normals = mesh.vertex_normals

# %%
#eval_utils.plot_pcd(mesh.vertices[mask, :], size=10000)
eval_utils.plot_pcd([verts[i, :, :] for i in range(npart)], size=10000)

# %%
"""
vv = mesh.vertices[mask, :]
vn = mesh.vertex_normals[mask, :]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vv)
pcd.normals = o3d.utility.Vector3dVector(vn)
pmesh, dv = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                      depth=5, linear_fit=False)

f, v = pypoisson.poisson_reconstruction(mesh.vertices[mask, :] / 2,
                                        mesh.vertex_normals[mask, :],
                                        depth=5)
# %%
mesh2 = trimesh.Trimesh(np.asarray(pmesh.vertices),
                        np.asarray(pmesh.triangles))
mesh2.show()

"""
