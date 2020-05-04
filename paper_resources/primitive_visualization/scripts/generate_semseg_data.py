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
import kaolin as kal

# %%
bae_shapenet_base_path = '/data/ugui0/kawana/ShapeNetBAENet/data'
occ_shapenet_base_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
out_dir_base = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetBAESemSeg'
dry_run = False


# %%
def realign(src, tgt):
    EPS = 1e-12
    # Compute the relative scaling factor and scale the src cloud.
    src_min = src.min(-2, keepdims=True)
    src_max = src.max(-2, keepdims=True)
    tgt_min = tgt.min(-2, keepdims=True)
    tgt_max = tgt.max(-2, keepdims=True)

    src = ((src - src_min) /
           (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min
    return src


# %%
splits = ['train', 'val', 'test']
synset_to_label = {
    "02691156": "airplane",
    "03001627": "chair",
    "03636649": "lamp",
    "04379243": "table",
}
label_to_synset = {v: k for k, v in synset_to_label.items()}

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
total = len(occset.intersection(baeset))
pbar = tqdm(total=total)
for split in splits:
    for class_path in occ_shapenet_class_paths:
        class_id = os.path.basename(class_path)
        class_name = synset_to_label[class_id]
        with open(os.path.join(class_path, '{}.lst'.format(split))) as f:
            lst = [os.path.join(class_id, s.strip()) for s in f.readlines()]
            model_id_union = list(set(lst).intersection(baeset))
            for class_id_model_id in model_id_union:
                model_id = os.path.basename(class_id_model_id)
                occ_model_path = os.path.join(class_path, model_id,
                                              'pointcloud.npz')
                bae_model_path = os.path.join(bae_shapenet_base_path,
                                              class_id + '_' + class_name,
                                              'points', model_id + '.txt')
                if dry_run:
                    assert os.path.exists(occ_model_path)
                    assert os.path.exists(bae_model_path), bae_model_path

                    pbar.update(1)
                    continue
                # P, dim
                occ_points = np.load(occ_model_path)['points']

                bae_table = np.loadtxt(bae_model_path, delimiter=' ')

                bae_points = bae_table[:, :3]

                aligned_bae_points = realign(bae_points, occ_points)
                labels = bae_table[:, 6]

                out_dir_path = os.path.join(
                    out_dir_base,
                    class_id,
                    model_id,
                )

                if not os.path.exists(out_dir_path):
                    os.makedirs(out_dir_path)

                out_path = os.path.join(out_dir_path,
                                        'bae_semseg_labeled_pointcloud')
                np.savez(out_path, points=aligned_bae_points, labels=labels)
                pbar.update(1)

# %%
"""
bae_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetBAESemSeg/02691156/41dca8046b0edc8f26360e1e29a956c7/bae_semseg_labeled_pointcloud'
bae_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetBAESemSeg/02691156/4f8952ff04d33784f64801ad2940cdd5/bae_semseg_labeled_pointcloud.npz'
occ_path = bae_path.replace('ShapeNetBAESemSeg', 'ShapeNet').replace(
    'bae_semseg_labeled_pointcloud', 'pointcloud')

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
"""
