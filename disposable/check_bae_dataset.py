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
allset_hdf5_name = '/data/ugui0/kawana/ShapeNetBAENet/data/02691156_airplane/02691156_vox.hdf5'
allset_dict = h5py.File(allset_hdf5_name, 'r')

# %%

synset_to_label = {
    "02691156": "airplane",
    "03001627": "chair",
    "03636649": "lamp",
    "04379243": "table",
}
label_to_synset = {v: k for k, v in synset_to_label.items()}

splits = ['train', 'val', 'test']
test_split = 'test'
bae_shapenet_base_path = '/data/ugui0/kawana/ShapeNetBAENet/data'
bae_shapenet_class_paths = [
    s for s in glob.glob(os.path.join(bae_shapenet_base_path, '*'))
    if os.path.basename(s).split('_')[0] in synset_to_label
]
all_models = []
test_models = []
for path in bae_shapenet_class_paths:
    class_id = os.path.basename(path).split('_')[0]

    with open(os.path.join(path, '{}_vox.txt'.format(class_id))) as f:
        lst = [s.strip() for s in f.readlines()]
    all_models.extend(lst)

    with open(os.path.join(path, '{}_test_vox.txt'.format(class_id))) as f:
        lst = [s.strip() for s in f.readlines()]
    test_models.extend(lst)
baeset = set(all_models)
test_baeset = set(test_models)
# %%
#02691156_airplane/02691156_{}_vox.txt'.format(
occ_shapenet_base_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
occ_shapenet_class_paths = [
    os.path.join(occ_shapenet_base_path, s) for s in synset_to_label
]

all_models = []
test_models = []
for split in splits:
    for class_path in occ_shapenet_class_paths:
        with open(os.path.join(class_path, '{}.lst'.format(split))) as f:
            lst = [s.strip() for s in f.readlines()]
        all_models.extend(lst)
        if split == 'test':
            test_models.extend(lst)
occset = set(all_models)
test_occset = set(test_models)

print('bae {} len:'.format('test'), len(test_baeset))
print('occ {} len:'.format('test'), len(test_occset))

print('occ - bae {} diff len'.format('test'), len(test_occset - test_baeset))

# %%
paths = glob.glob(
    '/data/ugui0/kawana/ShapeNetBAENet/data/02691156_airplane/points/*.txt')
labels = []
for path in paths:
    df = pd.read_csv(path, sep=' ')
    df.columns = ['x', 'y', 'z', 'a', 'b', 'c', 'label']
    labels.extend(df['label'].unique().tolist())
unique_labels = set(labels)
# %%
paths = glob.glob(
    '/data/ugui0/kawana/ShapeNetBAENet/data/02691156_airplane/points/*.txt')
labels = []
for path in paths:
    df = pd.read_csv(path, sep=' ')
    df.columns = ['x', 'y', 'z', 'a', 'b', 'c', 'label']
    if len(unique_labels) == len(df['label'].unique()):
        break

colors = ['blue', 'red', 'green', 'purple', 'orange']
plots = []
for idx, label in enumerate(df['label'].unique()):
    dflen = len(df[df['label'] == label])
    if dflen >= 100:
        select = random.sample(range(dflen), 100)
        x = df[df['label'] == label]['x'].iloc[select]
        y = df[df['label'] == label]['y'].iloc[select]
        z = df[df['label'] == label]['z'].iloc[select]
    else:
        x = df[df['label'] == label]['x']
        y = df[df['label'] == label]['y']
        z = df[df['label'] == label]['z']
    marker_opt = dict(color=colors[idx], size=1)
    plots.append(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                              marker=marker_opt))
# %%
import kaolin as kal
# %%
occ_shapenet_base_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
points = np.load(
    os.path.join(
        occ_shapenet_base_path,
        '02691156/cdb17eb7b14c83f225e27d5227712286/pointcloud.npz'))['points']
re_points = realign(points, df[['x', 'y', 'z']].to_numpy())
df2 = pd.DataFrame(re_points, columns=['x', 'y', 'z'])
select = random.sample(range(len(df2)), 100)
x2 = df2['x'].iloc[select]
y2 = df2['y'].iloc[select]
z2 = df2['z'].iloc[select]
marker_opt = dict(color='orange', size=1)
plots.append(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=marker_opt))
fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()


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
