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
import shutil
from datetime import datetime
import random
random.seed(0)
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
side_length_scale = 0.01

topk = 500
top_minus = 400
vis_num = 20
vis_count = 3
threshold = 0.90
max_holes = 100
min_holes = 3
holes_class = ['chair', 'table']
target = 'BSPNet_256'
ours = 'SHNet_30'
assert topk - top_minus >= 1

colormap_name = 'jet'
class_names = ['airplane', 'chair', 'table', 'rifle', 'car', 'bench']
#primitive_id_map = {'airplane': list(range(10)), 'chair': list(range(10))}

models_config_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/model_configs.yaml'
configs = yaml.load(open(models_config_path, 'r'))
attrs = configs[ours]

theirs_mesh_dirs = {
    'BSPNet_256':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808',
    'AtlasNetV2_30':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn30_20200503_221032/generation_explicit__20200503_221915',
    'OccNet':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403'
}
#theirs_mesh_dirs = {}

base_eval_dir = attrs['base_eval_dir']
fscore_pkl_path = attrs['fscore']
config_path = attrs['config_path']

rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
rendering_out_base_dir = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/shapenet_svr_comparison'
date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
rendering_out_dir = os.path.join(rendering_out_base_dir,
                                 'resources_{}'.format(date_str))
camera_param_path = os.path.join(rendering_out_base_dir, 'camera_param.txt')
ours_mesh_dirs = {
    'SHNet_30':
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812'
}
holes_count_pkl = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/shapenet_svr_comparison/resources/hole_count.pkl'
# %%
"""
good_list = [
    'table_f19fb80eae2ed668962a9d5e42f53a23',
    'table_e5567158ce772a43fcf7d910cd22d7c2',
    'table_e2dac2205ed98fad5067eac75a07f7',
    'table_d6e677600a93bc118ac5263758737a81',
    'table_d3fd6d332e6e8bccd5382f3f8f33a9f4',
    'table_cd94233033b1d958ef2438b4b778b7f8',
    'chair_dfc2328946b2d54a29426a0f57e4d15e',
    'chair_dfeb8d914d8b28ab5bb58f1e92d30bf7',
    'chair_d3302b7fa6504cab1a461b43b8f257f',
    'chair_d23682341fc187a570732116fb5f6e1',
    'chair_ed108ed496777cf6490ad276cd2af3a4',
    'chair_fd1a87a399c1c82d492d9da2668ec34c',
    'chair_d23682341fc187a570732116fb5f6e1',
    'car_fc521be0cb604c1aee4687e8f2543e',
    'car_ff267b1a6d986426c6df82b90873315e',
    'car_ef966d85be54c98ab002e5b0265e7e9d',
    'car_ed2e4dafc745bdd661fd7e090d4d0d45',
    'car_e3d7957c7a9b95382e877e82c90c24d',
    'bench_cb48fccedaee0e82368bd71100fb3a30',
    'bench_ce23a5781e86368af4fb4dee5181bee',
    'bench_f5557538f4c6d755d295b24579cf55b8',
    'airplane_f2171bb2d715140c8b96ae1a0a8b84ec',
    'airplane_dadf41579d385b0aacf77e718d93f3e1',
    'airplane_e8d5a3e98c222583d972c9dd75ed77d4',
    'airplane_dd9e42969d34463aca8607f540cc62ba',
    'rifle_e1fd9153928081d93b80701afa3beec5',
    'rifle_ff3425cf1860b6116314c3b6a3a65519',
    'rifle_d5734bfe7c57d3bda1bdbe5c0cfcf6e8',
    'rifle_cc362ac31993fcb4fa0d7d9af888ead',
    'table_d1ef95530469a1de1fc4857cc94b6562',
    'table_db96923291ea465d593ebeeedbff73b',
    'chair_dbfab57f9238e76799fc3b509229d3d',
]
"""
good_list = []
# %%
if not os.path.exists(rendering_out_dir):
    os.makedirs(rendering_out_dir)
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

wdf = pickle.load(open(configs[target]['fscore'], 'rb'))

hdf = pickle.load(open(holes_count_pkl, 'rb'))
hdf['class id'] = hdf['class_id']
# %%
# TODO: for loop here

fscore_key = 'fscore_th={} (mesh)'.format(side_length_scale)

gdf_f = gdf[['class id', 'modelname', fscore_key]]
wdf_f = wdf[['class id', 'modelname', fscore_key]]

gdf_f = gdf_f.rename(columns={fscore_key: 'gf1'})
wdf_f = wdf_f.rename(columns={fscore_key: 'wf1'})

merged = gdf_f.merge(wdf_f, on=['class id', 'modelname']).dropna()
merged = merged.merge(hdf, on=['class id', 'modelname']).dropna()
merged['diff'] = merged['gf1'] - merged['wf1']

# %%
samples_to_render = []
for idx in range(len(good_list)):
    l = good_list[idx]
    class_name, model_id = l.split('_')
    class_id = label_to_synset[class_name]
    samples_to_render.append((class_id, model_id))

if len(samples_to_render) == 0:
    for class_name_idx in range(len(class_names)):
        class_name = class_names[class_name_idx]
        class_id = label_to_synset[class_name]

        merged_class = merged[merged['class id'] ==
                              label_to_synset[class_name]]

        cond = merged_class['gf1'] > threshold
        cond = merged_class['diff'] > 0.1
        """
        if class_name in holes_class:
            cond = cond & (max_holes >= merged_class['holes']) & (
                merged_class['holes'] >= min_holes)
        """

        #topkdf = merged_class[cond].nlargest(top_minus, 'diff')
        topkdf = merged_class[cond].nlargest(top_minus, 'gf1')
        assert not len(topkdf) == 0
        random_idx = random.sample(range(len(topkdf)),
                                   min(vis_num, len(topkdf)))

        # TODO: for loop here
        cnt = 0
        for topk_idx in random_idx:
            """
        for topk_idx in range(topk):
            if topk_idx == 0 or topk_idx <= topk - top_minus or len(
                    topkdf) - 1 == topk_idx:
                cnt += 1
                model_id = topkdf['modelname'].iloc[topk_idx]

                samples_to_render.append((class_id, model_id))
                if len(topkdf) - 1 == topk_idx or cnt > vis_count:
                    break
            """
            if topk_idx >= len(topkdf):
                continue
            model_id = topkdf['modelname'].iloc[topk_idx]
            samples_to_render.append((class_id, model_id))

# %%
mesh_dirss = [theirs_mesh_dirs, ours_mesh_dirs]
filename_template = '{method}_{class_name}_{model_id}'
for class_id, model_id in samples_to_render:
    class_name = synset_to_label[class_id]
    # Render gt
    gt_model_path = os.path.join(shapenetv1_path, class_id, model_id,
                                 'model.obj')
    if not os.path.exists(gt_model_path):
        print('gt mesh not exists')
        continue
    with tempfile.NamedTemporaryFile(suffix='.off') as f:
        eval_utils.export_gt_mesh(gt_model_path, f.name)

        eval_utils.render_by_blender(rendering_script_path,
                                     camera_param_path,
                                     f.name,
                                     rendering_out_dir,
                                     filename_template.format(
                                         class_name=class_name,
                                         model_id=model_id,
                                         method='gt'),
                                     skip_reconvert=True,
                                     use_cycles=True,
                                     use_lamp=True)
    input_image = os.path.join(ours_mesh_dirs[ours], 'input', class_id,
                               model_id + '.jpg')
    shutil.copy(
        input_image,
        os.path.join(
            rendering_out_dir, 'input_{name}_{id}.jpg'.format(name=class_name,
                                                              id=model_id)))

    for mesh_dirs in mesh_dirss:
        for method_name, mesh_dir in mesh_dirs.items():
            mesh_path = os.path.join(mesh_dir, 'meshes', class_id,
                                     model_id + '.off')
            if not os.path.exists(mesh_path):
                print(method_name, 'no mesh')
                continue
            with tempfile.NamedTemporaryFile(suffix='.off') as f:
                mesh = trimesh.load(mesh_path)
                trimesh.repair.fix_inversion(mesh)
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fix_winding(mesh)

                eval_utils.render_by_blender(rendering_script_path,
                                             camera_param_path,
                                             mesh_path,
                                             rendering_out_dir,
                                             filename_template.format(
                                                 class_name=class_name,
                                                 model_id=model_id,
                                                 method=method_name),
                                             use_cycles=True,
                                             use_lamp=True)
