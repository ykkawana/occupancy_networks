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
splits = ['train', 'val', 'test']
synset_to_label = {
    "02691156": "airplane",
    "03001627": "chair",
    "03636649": "lamp",
    "04379243": "table",
}
label_to_synset = {v: k for k, v in synset_to_label.items()}

test_split = 'test'

# %%
bae_shapenet_class_paths = [
    s for s in glob.glob(os.path.join(bae_shapenet_base_path, '*'))
    if os.path.basename(s).split('_')[0] in synset_to_label
]
all_models = []
for path in bae_shapenet_class_paths:
    class_id = os.path.basename(path).split('_')[0]

    lst = [
        os.path.join(class_id,
                     os.path.basename(s).split('.')[0])
        for s in glob.glob(os.path.join(path, 'points', '*.txt'))
    ]
    all_models.extend(lst)

baeset = set(all_models)
# %%
occ_shapenet_class_paths = [
    os.path.join(occ_shapenet_base_path, s) for s in synset_to_label
]

trainval_models = []
test_models = []
for split in splits:
    for class_path in occ_shapenet_class_paths:
        class_id = os.path.basename(class_path)
        with open(os.path.join(class_path, '{}.lst'.format(split))) as f:
            lst = [os.path.join(class_id, s.strip()) for s in f.readlines()]
        if split == 'test':
            test_models.extend(lst)
        else:
            trainval_models.extend(lst)
trainval_occset = set(trainval_models)
test_occset = set(test_models)

# %%
occset_dict = {'test': test_occset, 'trainval': trainval_occset}
for split, split_occset in occset_dict.items():
    inter = split_occset.intersection(baeset)
    class_dict = defaultdict(lambda: [])
    for class_id_model_id in inter:
        class_id, model_id = class_id_model_id.split('/')
        class_dict[class_id].append(model_id)

    for class_id, model_ids in class_dict.items():
        txt = '\n'.join(model_ids)
        out_path = os.path.join(out_dir_base, class_id, split + '.lst')
        if dry_run:
            print(txt)
            print(out_path)
        else:
            with open(out_path, 'w') as f:
                f.write(txt)
