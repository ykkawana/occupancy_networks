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
from kaolin.transforms import pointcloudfunc as pcfunc
os.chdir('/home/mil/kawana/workspace/occupancy_networks')

# %%
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization'
rendering_out_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/resources_20200511_internal_review_copied_20200518'
camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')
mesh_path = '/home/mil/kawana/workspace/superquadric_parsing/scripts/output/primitives.obj'
class_id = '02691156'
model_id = 'fe0eb72a9fb21dd62b600da24e0965'
colormap_name = 'jet'
angle = -90
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


def roty90(points_iou, angle):
    if isinstance(points_iou, np.ndarray):
        points_iou = torch.from_numpy(points_iou).float()
    rad = float(angle) / 180. * np.pi
    roty = [[np.cos(rad), 0., np.sin(rad)], [0., 1., 0.],
            [-np.sin(rad), 0., np.cos(rad)]]

    rotm = torch.tensor(roty).float()
    return pcfunc.rotate(points_iou, rotm).numpy()


# %%
class_name = synset_to_label[class_id]
mesh = trimesh.load(mesh_path)
mesh = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy())

verts = mesh.vertices
verts = roty90(verts, angle)

primitive_verts_for_rendering = eval_utils.normalize_verts_in_occ_way(verts)
mesh.vertices = primitive_verts_for_rendering

meshes = mesh.split()
nparts = len(meshes)
bspnet_primitive_id_map = {'airplane': list(range(nparts))}
cm = plt.get_cmap(colormap_name, nparts)
# normed to 0 - 1
rgbas = [np.array(cm(idx)) for idx in range(nparts)]

colors = []
colors_emphasis = []
for pidx, rgba in enumerate(rgbas):

    colors.append(rgba)
    colors_emphasis.append(rgba)

filename_template = 'arbitralmesh_{class_name}_{model_id}_{type}_{id}'
with tempfile.TemporaryDirectory() as dname:
    dname = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/cache'

    model_name = 'colored_mesh'
    model_path = os.path.join(dname, '{}.obj'.format(model_name))

    eval_utils.export_colored_mesh(meshes, colors, model_path)

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 model_path,
                                 rendering_out_dir,
                                 filename_template.format(
                                     class_name=class_name,
                                     model_id=model_id,
                                     type=model_name,
                                     id=0),
                                 skip_reconvert=True)

    model_name = 'emphasis_mesh'
    model_path = os.path.join(dname, '{}.obj'.format(model_name))

    eval_utils.export_colored_mesh(meshes, colors_emphasis, model_path)

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 model_path,
                                 rendering_out_dir,
                                 filename_template.format(
                                     class_name=class_name,
                                     model_id=model_id,
                                     type=model_name,
                                     id=0),
                                 skip_reconvert=True)

    for pidx, (mesh, color) in enumerate(zip(meshes, colors)):
        if not pidx in bspnet_primitive_id_map[synset_to_label[class_id]]:
            continue
        model_name = 'colored_primitive_mesh'
        model_path = os.path.join(dname, '{}.obj'.format(model_name))

        mesh.vertices = eval_utils.normalize_verts_in_occ_way(mesh.vertices)
        eval_utils.export_colored_mesh([mesh], [color], model_path)

        eval_utils.render_by_blender(rendering_script_path,
                                     camera_param_path,
                                     model_path,
                                     rendering_out_dir,
                                     filename_template.format(
                                         class_name=class_name,
                                         model_id=model_id,
                                         type=model_name,
                                         id=pidx),
                                     skip_reconvert=True)
