# %%
import kaolin as kal
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
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
import dotenv
dotenv.load_dotenv(verbose=True)
from tqdm import tqdm

from joblib import Parallel, delayed
import pandas as pd
# %%
occ_shapenet_root = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
out_path = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/shapenet_svr_comparison/resources/hole_count.pkl'
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
label_to_kal_label = {
    'vessel': 'watercraft',
    'loudspeaker': 'speaker',
    'display': 'monitor',
    'airplane': 'plane',
    'telephone': 'phone',
}
cache_root = os.getenv('SHAPENET_KAOLIN_RES_256_CACHE_ROOT')
shapenet_root = os.getenv('SHAPENET_ROOT')
label_to_synset = {v: k for k, v in synset_to_label.items()}

category = 'plane'
categories = [category]
cache_dir = os.path.join(cache_root, category)

liness = []
for class_id, class_name in synset_to_label.items():
    kal_class_name = label_to_kal_label.get(class_name, class_name)
    path = os.path.join(cache_root, kal_class_name, 'surface_meshes',
                        '2a86b3e0a00bf3cfa3a3f48670f6a079')
    if not os.path.exists(path):
        print(class_name)
    class_list = os.path.join(occ_shapenet_root, class_id, 'test.lst')
    with open(class_list) as f:
        lines = [(path, class_id, line.strip()) for line in f.readlines()]
    liness.extend(lines)

num_proc = 30
indices = np.random.permutation(np.arange(len(liness)))
indices_list = np.array_split(indices, num_proc)


def get_eu(inputs):
    indices, lines = inputs
    dis = []
    for idx in tqdm(indices):
        path, class_id, modelname = lines[idx]
        ppath = os.path.join(path, modelname + '.p')
        if not os.path.exists(ppath):
            continue
        p = torch.load(ppath)
        v = p['vertices']
        f = p['faces']
        m = trimesh.Trimesh(v.numpy(), f.numpy())
        eu = (2 - (len(m.vertices) - len(m.edges_unique) + len(m.faces))) / 2
        dis.append({
            'modelname': modelname,
            'class_id': class_id,
            'holes': int(eu)
        })

    return dis


r = Parallel(n_jobs=num_proc)(
    [delayed(get_eu)((indices, liness)) for indices in indices_list])

all_dicts = []
for dicts in r:
    all_dicts.extend(dicts)

df = pd.DataFrame(all_dicts)
df.to_pickle(out_path)
