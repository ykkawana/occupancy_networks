# %%
import h5py
import os
import numpy as np
import glob
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import random

# %%
bae_shapenet_base_path = '/data/ugui0/kawana/ShapeNetBAENet/data'
occ_shapenet_base_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
bsp_shapenet_base_path = '/home/mil/kawana/workspace/occupancy_networks/external/bspnet/data/all_vox256_img'
# %%
synset_to_label = {
    "02691156": "airplane",
    "03001627": "chair",
    "03636649": "lamp",
    "04379243": "table",
}
label_to_synset = {v: k for k, v in synset_to_label.items()}
# %%
bsp_list_name_template = 'all_vox256_img_{}.txt'
bsp_hdf5_name_template = 'all_vox256_img_{}.hdf5'
all_models = []
bsp_data = {}
splits = ['train', 'test']
for split in splits:
    with open(
            os.path.join(bsp_shapenet_base_path,
                         bsp_list_name_template.format(split))) as f:
        lst = [s.strip() for s in f.readlines()]
    all_models.extend(lst)
    bsp_data[split] = h5py.File(
        os.path.join(bsp_shapenet_base_path,
                     bsp_hdf5_name_template.format(split)), 'r')
    if split == 'test':
        bsp_test_models = lst
    if split == 'train':
        bsp_train_models = lst
bspset = set(all_models)
test_bspset = set(bsp_test_models)


def get_bsp_data(class_model_id):
    if class_model_id in bsp_test_models:
        idx = bsp_test_models.index(class_model_id)
        data = bsp_data['test']
    else:
        idx = bsp_train_models.index(class_model_id)
        data = bsp_data['train']

    return {key: item[idx, ...] for key, item in bsp_data.items()}


# %%
splits = ['train', 'val', 'test']

test_split = 'test'
bae_shapenet_class_paths = [
    s for s in glob.glob(os.path.join(bae_shapenet_base_path, '*'))
    if os.path.basename(s).split('_')[0] in synset_to_label
]
all_models = []
for path in bae_shapenet_class_paths:
    class_id = os.path.basename(path).split('_')[0]

    #with open(os.path.join(path, '{}_vox.txt'.format(class_id))) as f:
    #    lst = [s.strip() for s in f.readlines()]
    lst = [
        os.path.join(class_id,
                     os.path.basename(s).split('.')[0])
        for s in glob.glob(os.path.join(path, 'points', '*.txt'))
    ]
    all_models.extend(lst)

baeset = set(all_models)
# %%
#02691156_airplane/02691156_{}_vox.txt'.format(
occ_shapenet_class_paths = [
    os.path.join(occ_shapenet_base_path, s) for s in synset_to_label
]

all_models = []
for split in splits:
    for class_path in occ_shapenet_class_paths:
        class_id = os.path.basename(class_path)
        with open(os.path.join(class_path, '{}.lst'.format(split))) as f:
            lst = [os.path.join(class_id, s.strip()) for s in f.readlines()]
        all_models.extend(lst)
occset = set(all_models)

# %%
print('models not in occ but in bae', len(baeset - occset))
print('models not in bae but in occ', len(occset - baeset))

# %%
inter = occset.intersection(baeset).intersection(bspset)
inter_list = list(inter)
# %%
class_id, model_id = inter_list[0].split('/')

occ_path = os.path.join(occ_shapenet_base_path, class_id, model_id,
                        'pointcloud.npz')
occ_points = np.load(occ_path)['points']

bae_path = os.path.join(bae_shapenet_base_path,
                        class_id + '_' + synset_to_label[class_id], 'points',
                        model_id + '.txt')
bae_points = np.loadtxt(bae_path, delimiter=' ')[:3]

bsp_points_occ = get_bsp_data(inter_list[0])['values_64']
bsp_points_all = get_bsp_data(inter_list[0])['points_64']

points_num = 1000

bae_points = np.load(bae_path)['points']
selected_bae_points = bae_points[
    random.sample(range(len(bae_points)), points_num), :]
occ_points = np.load(occ_path)['points']
selected_occ_points = occ_points[
    random.sample(range(len(occ_points)), points_num), :]

plots = []
plots.append(
    go.Scatter3d(x=selected_bae_points[:, 0],
                 y=selected_bae_points[:, 1],
                 z=selected_bae_points[:, 2],
                 mode='markers',
                 marker=dict(color='blue', size=1)))
plots.append(
    go.Scatter3d(x=selected_occ_points[:, 0],
                 y=selected_occ_points[:, 1],
                 z=selected_occ_points[:, 2],
                 mode='markers',
                 marker=dict(color='red', size=1)))
fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()
