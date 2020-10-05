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
from matplotlib import colors as plt_colors
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
import eval_utils
import skimage.io
from PIL import Image
import shutil

os.chdir('/home/mil/kawana/workspace/occupancy_networks')
dotenv.load_dotenv('/home/mil/kawana/workspace/occupancy_networks/.env',
                   verbose=True)

date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))


def separate_mesh_and_color(mesh_color_list):
    meshes = [mesh for mesh, _ in mesh_color_list]
    colors = [color for _, color in mesh_color_list]

    return meshes, colors


# %%
skip_primitive_rendering = True
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
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/gen_config__20200502_003123.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/gen_config__20200502_003123.yaml'
config_path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153/gen_config__20200502_103153.yaml'
render_modelname = 'fe0eb72a9fb21dd62b600da24e0965'
render_class = 'airplane'
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/overview'
rendering_out_dir = os.path.join(rendering_out_base_dir,
                                 'resources_{}'.format(date_str))
if not os.path.exists(rendering_out_dir):
    os.makedirs(rendering_out_dir)
camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')

mesh_gen_path_rendering = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit__20200502_043815'
mesh_gen_path_rendering = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153'
mesh_gen_path_rendering = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200502_041512/generation_explicit__20200502_103153'
mesh_gen_path = os.path.dirname(config_path)
base_eval_dir = os.path.dirname(mesh_gen_path)
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

#cfg['model']['decoder_kwargs']['extract_surface_point_by_max'] = True

if debug:
    #cfg['data']['debug'] = {'sample_n': 50}
    cfg['data']['classes'] = ['02691156']

cfg['test']['is_eval_explicit_mesh'] = True
cfg['generation']['is_explicit_mesh'] = True
cfg['data']['is_normal_icosahedron'] = True
cfg['data']['icosahedron_subdiv'] = 4
cfg['trainer']['is_get_radius_direction_as_normals'] = True
#eval_utils.update_dict_with_options(cfg, unknown_args)

is_cuda = True
device = torch.device("cuda" if is_cuda else "cpu")

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
dataset.fields['patch'] = data.PlanarPatchField(
    'test', cfg['data'].get('patch_side_length', 200), True)
indices = {}
for idx in range(len(dataset)):
    model_dict = dataset.get_model_dict(idx)
    model_id = model_dict['model']
    class_id = model_dict.get('category', 'n/a')
    indices[(class_id, model_id)] = idx
#

# Model

model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(base_eval_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generate
model.eval()

is_pnet = cfg['method'] == 'pnet'

if is_pnet:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
else:
    raise NotImplementedError

if 'n_primitives' in cfg['model']['decoder_kwargs']:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
cm = plt.get_cmap('jet', nparts)

# normed to 0 - 1
rgbas = [np.array(cm(idx)) for idx in range(nparts)]
# %%
render_class_id = label_to_synset[render_class]
data = dataset[indices[(render_class_id, render_modelname)]]

# %%
# Render colored primitives
model_dict = dataset.get_model_dict(idx)
model_id = model_dict['model']
class_id = model_dict.get('category', 'n/a')
class_name = synset_to_label[class_id]

normal_angles = data.get('angles.normal_angles').to(device).unsqueeze(0)
normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
inputs = data.get('inputs').to(device).unsqueeze(0)
feature = model.encode_inputs(inputs)

kwargs = {}
point_scale = cfg['trainer']['pnet_point_scale']

#scaled_coord = points * point_scale
with torch.no_grad():
    output = model.decode(None, None, feature, angles=normal_angles, **kwargs)
    normal_vertices, _, _, _, _ = output

B, N, P, D = normal_vertices.shape

# TODO: for loop here
colored_meshes = []
emphasis_meshes = []
colored_primitives = {}
colored_primitives2 = {}
for pidx in range(N):
    verts = (normal_vertices[0, pidx, :, :] /
             point_scale).to('cpu').detach().numpy()
    faces = normal_faces[0, :, :].to('cpu').detach().numpy()
    primitive_verts_for_rendering = eval_utils.normalize_verts_in_occ_way(
        verts)

    rgba = rgbas[pidx]

    # For primitive wise rendering
    mesh = trimesh.Trimesh(verts, faces, process=False)
    colored_meshes.append((mesh, rgba))

    primitive_mesh = trimesh.Trimesh(primitive_verts_for_rendering,
                                     faces,
                                     process=False)
    primitive_mesh2 = trimesh.Trimesh(verts, faces, process=False)
    colored_primitives[pidx] = (primitive_mesh, rgba)
    colored_primitives2[pidx] = (primitive_mesh2, rgba)

filename_template = '{class_name}_{model_id}_{type}_{id}'

if not skip_primitive_rendering:
    with tempfile.TemporaryDirectory() as dname:
        dname = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/cache'

        model_name = 'colored_mesh'
        model_path = os.path.join(dname, '{}.obj'.format(model_name))

        meshes, colors = separate_mesh_and_color(colored_meshes)
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

        for pidx, (mesh, color) in colored_primitives.items():
            model_name = 'colored_primitive_mesh'
            model_path = os.path.join(dname, '{}.obj'.format(model_name))

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

        for pidx, (mesh, color) in colored_primitives2.items():
            model_name = 'colored_primitive_mesh_original_size'
            model_path = os.path.join(dname, '{}.obj'.format(model_name))

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

# %%
# Render mesh
mesh_path = os.path.join(mesh_gen_path_rendering, 'meshes', render_class_id,
                         render_modelname + '.off')
eval_utils.render_by_blender(rendering_script_path, camera_param_path,
                             mesh_path, rendering_out_dir, 'mesh')
#       use_cycles=True,
#       use_lamp=True)

# %%
# Render pointcloud
pointcloud = data.get('pointcloud')
sp_num = 1000
rnd_idx = np.random.choice(range(pointcloud.shape[0]), size=sp_num)
point_selected = pointcloud[rnd_idx, :]
sp_scale = 0.008
spheres = [
    trimesh.creation.icosphere(subdivisions=2, radius=sp_scale)
    for _ in range(sp_num)
]
for point, sphere in zip(point_selected, spheres):
    sphere.vertices = (sphere.vertices + point)
point_mesh = trimesh.util.concatenate(spheres)
with tempfile.NamedTemporaryFile(suffix='.obj') as f:
    eval_utils.export_colored_mesh([point_mesh],
                                   [(0.368417, 0.506779, 0.709798, 1.0)],
                                   f.name)

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 f.name,
                                 rendering_out_dir,
                                 'pointcloud',
                                 skip_reconvert=True)

# %%
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
    coord = ((data.get('patch.mesh_vertices').unsqueeze(0).float() - 0.5) *
             1.05) * point_scale
    coord1 = torch.cat(
        [coord,
         torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1)],
        axis=-1).to(device)
    #coord = torch.cat([coord[:, :, 0].unsqueeze(-1), torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1), coord[:, :, 1].unsqueeze(-1) ], axis=-1).to(device)

    output = model(coord1, inputs, sample=True, angles=angles, **kwargs)
vertices, vertices_mask, sgd, _, _ = output

si = torch.tanh(sgd.max(1)[0].view(200, 200) * 3)

cm = plt.get_cmap('coolwarm', 256)

wb = (si.cpu().numpy() + 1) / 2 * 255
wb = np.flipud(wb.astype(np.uint8))
wb = wb.reshape(200 * 200)
heatmap = np.zeros([200 * 200, 3], dtype=np.uint8)
for c in range(256):
    color = np.array(cm(c)[:3]) * 255
    color = color.astype(np.uint8)
    heatmap[wb == c, :] = color

out_image_path = os.path.join(rendering_out_dir, 'indicator_function1.png')
heatmap1 = heatmap.reshape(200, 200, 3)
skimage.io.imsave(out_image_path, heatmap1)

with torch.no_grad():
    coord = ((data.get('patch.mesh_vertices').unsqueeze(0).float() - 0.5) *
             1.05) * point_scale
    #coord = torch.cat([coord, torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1)], axis=-1).to(device)
    coord2 = torch.cat([
        coord[:, :, 0].unsqueeze(-1),
        -0.29 + torch.zeros([1, 1, 1]).float().expand(-1, coord.shape[1], -1),
        coord[:, :, 1].unsqueeze(-1)
    ],
                       axis=-1).to(device)

    output = model(coord2, inputs, sample=True, angles=angles, **kwargs)
vertices, vertices_mask, sgd, _, _ = output

si = torch.tanh(sgd.max(1)[0].view(200, 200) * 3)

cm = plt.get_cmap('coolwarm', 256)

wb = (si.cpu().numpy() + 1) / 2 * 255
wb = wb.astype(np.uint8).reshape(200 * 200)
heatmap = np.zeros([200 * 200, 3], dtype=np.uint8)
for c in range(256):
    color = np.array(cm(c)[:3]) * 255
    color = color.astype(np.uint8)
    heatmap[wb == c, :] = color
out_image_path = os.path.join(rendering_out_dir, 'indicator_function2.png')
heatmap2 = heatmap.reshape(200, 200, 3)
skimage.io.imsave(out_image_path, heatmap2)
"""
heatmap1 = np.concatenate([heatmap1, np.ones([200, 200, 1])], axis=-1)
heatmap1[:67, :, 3] = 0
heatmap1[125:, :, 3] = 0

heatmap2 = np.concatenate([heatmap2, np.ones([200, 200, 1])], axis=-1)
heatmap2[:20, :, 3] = 0
heatmap2[175:, :, 3] = 0
"""

heatmap_all = np.concatenate([heatmap1, heatmap2], axis=0)
out_image_path = os.path.join(rendering_out_dir, 'indicator_texture.png')
with tempfile.NamedTemporaryFile(suffix='.png') as f:
    #skimage.io.imsave(f.name, heatmap_all)
    #pil_image = Image.open(f.name)
    pil_image = Image.fromarray(heatmap_all)
    pil_image.save(f.name)
    pil_image = Image.open(f.name)

coord1_np = coord1.cpu().numpy()[0] / point_scale
coord1_np[:, 1] = coord1_np[:, 1] + 0.008
coord2_np = coord2.cpu().numpy()[0] / point_scale
coord1_np[:, 1] = np.clip(coord1_np[:, 1], -20. / 100., 20. / 100.)
coord2_np[:, 2] = np.clip(coord2_np[:, 2], -40. / 100., 40. / 100.)
verts_all = np.concatenate([coord1_np, coord2_np], axis=0)
face1 = data.get('patch.mesh_faces').numpy().copy()
face2 = data.get('patch.mesh_faces').numpy().copy() + len(coord1_np)
face_all = np.concatenate([face1, face2], axis=0)
uv1 = data.get('patch.mesh_vertices').numpy().copy()
uv1[:, 1] = uv1[:, 1] / 2
uv2 = data.get('patch.mesh_vertices').numpy().copy()
uv2[:, 1] = np.flipud(uv2[:, 1] / 2 + 0.5)
uvs = np.concatenate([uv2, uv1], axis=0)
texture = trimesh.visual.TextureVisuals(uv=uvs, image=pil_image)
mesh_all = trimesh.Trimesh(verts_all, face_all, visual=texture)
text, tex_data = trimesh.exchange.obj.export_obj(mesh_all,
                                                 include_texture=True)

with tempfile.TemporaryDirectory() as indicator_out_dir:
    #indicator_out_dir = os.path.join(rendering_out_dir, 'indicator_mesh')
    if not os.path.exists(indicator_out_dir):
        os.makedirs(indicator_out_dir)

    with open(os.path.join(indicator_out_dir, 'indicator.obj'), 'w') as f:
        f.write(text)

    mtl_name = 'material0.mtl'
    with open(os.path.join(indicator_out_dir, mtl_name), 'w') as f:
        f.write((tex_data[mtl_name]).decode('utf-8'))
        f.write('\nd 0.8')
        #f.write(bytearray(tex_data[mtl_name]))
    texture_name = 'material0.png'
    with open(os.path.join(indicator_out_dir, texture_name), 'wb') as f:
        f.write(bytearray(tex_data[texture_name]))

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 os.path.join(indicator_out_dir,
                                              'indicator.obj'),
                                 rendering_out_dir,
                                 'indicator',
                                 skip_reconvert=True)

# %%
# Texture
with tempfile.TemporaryDirectory() as fname:
    #dname = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/cache'
    texture_file = 'uvchecker.png'
    shutil.copy(
        '/home/mil/kawana/workspace/occupancy_networks/paper_resources/overview/resources/{}'
        .format(texture_file), fname)
    pil_image2 = Image.open(os.path.join(fname, texture_file))

    uvs = data.get('angles.normal_angles').numpy().copy()
    uvs[:, 0] = (uvs[:, 0] + np.pi) / (2 * np.pi)
    uvs[:, 1] = (uvs[:, 1] + np.pi / 2) / (np.pi)
    meshes, _ = separate_mesh_and_color(colored_meshes)
    face_all = np.concatenate(
        [mesh.faces + idx * len(uvs) for idx, mesh in enumerate(meshes)],
        axis=0)
    verts_all = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    uvs_all = []
    cnt = 0
    Y = 4
    X = 3
    for y in range(Y):
        for x in range(X):
            cnt += 1
            if cnt > len(meshes):
                break
            uv = uvs.copy()
            uv[:, 0] = uv[:, 0] / Y + 1 / Y * y
            uv[:, 1] = uv[:, 1] / X + 1 / X * x
            uvs_all.append(uv)

    #uvs_all = np.concatenate(uvs_all, axis=0)
    uvs_all = np.concatenate([uvs] * len(meshes), axis=0)

    texture2 = trimesh.visual.TextureVisuals(uv=uvs_all, image=pil_image2)

    textured_mesh = trimesh.Trimesh(verts_all, face_all, visual=texture2)
    text2, tex_data2 = trimesh.exchange.obj.export_obj(textured_mesh,
                                                       include_texture=True)

    #indicator_out_dir = os.path.join(rendering_out_dir, 'indicator_mesh')

    with open(os.path.join(fname, 'textured.obj'), 'w') as f:
        f.write(text2.replace('material0', 'material1'))

    mtl_name = 'material0.mtl'
    with open(os.path.join(fname, mtl_name.replace('material0', 'material1')),
              'w') as f:
        f.write((tex_data2[mtl_name]).decode('utf-8').replace(
            'material0', 'material1'))
        #f.write(bytearray(tex_data[mtl_name]))
    texture_name = 'material0.png'
    with open(
            os.path.join(fname, texture_name.replace('material0',
                                                     'material1')), 'wb') as f:
        f.write(bytearray(tex_data2[texture_name]))

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 os.path.join(fname, 'textured.obj'),
                                 rendering_out_dir,
                                 'textured',
                                 skip_reconvert=True)

# %%
# Normals
normal_angles.requires_grad = True
output = model.decode(None, None, feature, angles=normal_angles, **kwargs)
normal_vertices, _, _, _, radius = output
B, N, P, D = normal_vertices.shape

thetags = []
phigs = []
theta = normal_angles[..., 0]
phi = normal_angles[..., 1]
for idx in range(N):
    r = radius[:, idx, :]
    rg = torch.autograd.grad(
        r,
        normal_angles,
        torch.ones_like(r),
        retain_graph=True,
    )[0]
    thetag = torch.stack([
        rg[..., 0] * theta.cos() * phi.cos() - theta.sin() * phi.cos() * r,
        rg[..., 0] * theta.sin() * phi.cos() + theta.cos() * phi.cos() * r,
        rg[..., 0] * phi.sin()
    ],
                         axis=-1)
    thetags.append(thetag)
    phig = torch.stack([
        rg[..., 1] * theta.cos() * phi.cos() - theta.cos() * phi.sin() * r,
        rg[..., 1] * theta.sin() * phi.cos() - theta.sin() * phi.sin() * r,
        rg[..., 1] * phi.sin() + phi.cos() * r
    ],
                       axis=-1)
    phigs.append(phig)
thetags = torch.cat(thetags, axis=1)
phigs = torch.cat(phigs, axis=1)
uv_normals = torch.cross(thetags, phigs)

# %%
colors = eval_utils.normals2rgb(uv_normals[0, :,
                                           ...].detach().cpu().numpy().reshape(
                                               -1, 3),
                                append_dummpy_alpha=False)

verts = (normal_vertices[0, ...] / 6).to('cpu').detach().numpy()
npart, verts_n, _ = verts.shape
verts_all = verts.reshape([-1, 3])
faces = normal_faces[0, ...].to('cpu').detach().numpy()
faces_all = np.concatenate([faces + verts_n * i for i in range(npart)])

# For primitive wise rendering
mesh = trimesh.Trimesh(verts_all, faces_all, process=False)
mesh.visual.vertex_colors = colors

with tempfile.NamedTemporaryFile(suffix='.png') as f:
    #skimage.io.imsave(f.name, heatmap_all)
    #pil_image = Image.open(f.name)
    dest = f.name
    dest = os.path.join(rendering_out_dir, 'n.png')
    img = np.expand_dims(255 * colors, axis=1).astype(np.uint8)

    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((10, colors.shape[0]))
    pil_image.save(dest)
    pil_image = Image.open(dest)
uvs = np.ones([colors.shape[0], 2]).astype(np.float32) * 0.2
uvs[:, 1] = np.flipud(
    np.arange(colors.shape[0]).astype(np.float32) / colors.shape[0])
uvs[-1, 1] = 1.
texture = trimesh.visual.TextureVisuals(uv=uvs, image=pil_image)
mesh_all = trimesh.Trimesh(verts_all, face_all, visual=texture)
text, tex_data = trimesh.exchange.obj.export_obj(mesh_all,
                                                 include_texture=True)

with tempfile.TemporaryDirectory() as indicator_out_dir:
    indicator_out_dir = os.path.join(rendering_out_dir, 'normal_mesh')
    if not os.path.exists(indicator_out_dir):
        os.makedirs(indicator_out_dir)

    with open(os.path.join(indicator_out_dir, 'indicator.obj'), 'w') as f:
        f.write(text)

    mtl_name = 'material0.mtl'
    with open(os.path.join(indicator_out_dir, mtl_name), 'w') as f:
        f.write((tex_data[mtl_name]).decode('utf-8'))
        f.write('\nd 0.8')
        #f.write(bytearray(tex_data[mtl_name]))
    texture_name = 'material0.png'
    with open(os.path.join(indicator_out_dir, texture_name), 'wb') as f:
        f.write(bytearray(tex_data[texture_name]))

    eval_utils.render_by_blender(rendering_script_path,
                                 camera_param_path,
                                 os.path.join(indicator_out_dir, 'color2.obj'),
                                 rendering_out_dir,
                                 'normals',
                                 skip_reconvert=True)

# %%
