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
os.chdir('/home/mil/kawana/workspace/occupancy_networks')


# %%
def separate_mesh_and_color(mesh_color_list):
    meshes = [mesh for mesh, _ in mesh_color_list]
    colors = [color for _, color in mesh_color_list]

    return meshes, colors


# %%
shapenetv1_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v1'
shapenetv2_path = '/data/unagi0/kawana/workspace/ShapeNetCore.v2'
shapenetocc_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
side_length_scale = 0.0107337006427915

topk = 50
colormap_name = 'jet'
class_names = ['airplane', 'chair']
primitive_id_map = {'airplane': [0, 15, 29], 'chair': [0, 15, 29]}
base_eval_dir = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954'
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization'
rendering_out_dir = os.path.join(rendering_out_base_dir, 'resources')
camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')
mesh_dir_path = os.path.join(base_eval_dir, 'generation_explicit')
fscore_pkl_path = os.path.join(mesh_dir_path,
                               'eval_fscore_from_meshes_full_explicit.pkl')
config_path = os.path.join(base_eval_dir, 'config.yaml')

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
cfg['data']['classes'] = [label_to_synset[label] for label in class_names]
cfg['data']['is_normal_icosahedron'] = True
cfg['data']['icosahedron_subdiv'] = 4
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
if 'n_primitives' in cfg['model']['decoder_kwargs']:
    nparts = cfg['model']['decoder_kwargs']['n_primitives']
cm = plt.get_cmap(colormap_name, nparts)

# normed to 0 - 1
rgbas = [np.array(cm(idx)) for idx in range(nparts)]
gdf = pickle.load(open(fscore_pkl_path, 'rb'))

# %%
# TODO: for loop here

samples_to_render = []
for class_name_idx in range(len(class_names)):
    class_name = class_names[class_name_idx]
    class_id = label_to_synset[class_name]
    gdf_class = gdf[gdf['class id'] == label_to_synset[class_name]]

    fscore_key = 'fscore_th={} (mesh)'.format(side_length_scale)

    topkdf = gdf_class.nlargest(topk, fscore_key)

    # TODO: for loop here
    for topk_idx in range(topk):
        if topk_idx == 0 or topk_idx == topk - 1:
            model_id = topkdf['modelname'].iloc[topk_idx]

            samples_to_render.append((class_id, model_id))

indices = []
for idx in range(len(dataset)):
    model_dict = dataset.get_model_dict(idx)
    model_id = model_dict['model']
    class_id = model_dict.get('category', 'n/a')
    if (class_id, model_id) in samples_to_render:
        indices.append(idx)
# %%
# TODO: for loop here
for idx in indices:
    data = dataset[idx]
    model_dict = dataset.get_model_dict(idx)
    model_id = model_dict['model']
    class_id = model_dict.get('category', 'n/a')
    class_name = synset_to_label[class_id]

    assert (class_id, model_id) in samples_to_render

    print('found good one', synset_to_label[class_id], model_id)
    normal_angles = data.get('angles.normal_angles').to(device).unsqueeze(0)
    normal_faces = data.get('angles.normal_face').to(device).unsqueeze(0)
    inputs = data.get('inputs').to(device).unsqueeze(0)
    feature = model.encode_inputs(inputs)

    kwargs = {}
    point_scale = cfg['trainer']['pnet_point_scale']

    #scaled_coord = points * point_scale
    with torch.no_grad():
        output = model.decode(None,
                              None,
                              feature,
                              angles=normal_angles,
                              **kwargs)
        normal_vertices, _, _, _, _ = output

    B, N, P, D = normal_vertices.shape

    # TODO: for loop here
    colored_meshes = []
    emphasis_meshes = []
    colored_primitives = {}
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
        colored_primitives[pidx] = (primitive_mesh, rgba)

        emphasis_mesh = trimesh.Trimesh(verts, faces, process=False)
        if pidx in primitive_id_map[synset_to_label[class_id]]:
            rgba_tr = rgba
        else:
            rgba_tr = [0.5, 0.5, 0.5, 0.2]
        emphasis_meshes.append((emphasis_mesh, rgba_tr))

    filename_template = '{class_name}_{model_id}_{type}_{id}'
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

        model_name = 'emphasis_mesh'
        model_path = os.path.join(dname, '{}.obj'.format(model_name))

        meshes, colors = separate_mesh_and_color(emphasis_meshes)
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
            if not pidx in primitive_id_map[synset_to_label[class_id]]:
                continue
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
# %%
