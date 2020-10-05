import sys
#sys.path.insert(0, '..')
import kaolin as kal
from kaolin.datasets import shapenet
from kaolin import rep
from kaolin import conversions
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import utils
from losses import custom_chamfer_loss
import numpy as np
import random
import tqdm
from collections import defaultdict
from torch.autograd import Variable
import torch.optim as optim
import pickle
import os
import dotenv
import plotly.graph_objects as go
import eval_utils

# %%
seed = 0
random.seed(seed)
np.random.seed(seed)
# PyTorch のRNGを初期化
torch.manual_seed(seed)

# %%
dotenv.load_dotenv(verbose=True)

category = 'plane'
cache_root = os.getenv('SHAPENET_KAOLIN_CACHE_ROOT')
shapenet_root = os.getenv('SHAPENET_ROOT')
cache_dir = os.path.join(cache_root, category)

categories = [category]

# %%
sdf_set = shapenet.ShapeNet_SDF_Points(root=shapenet_root,
                                       categories=categories,
                                       cache_dir=cache_dir,
                                       train=True,
                                       split=1.)
point_set = shapenet.ShapeNet_Points(root=shapenet_root,
                                     categories=categories,
                                     cache_dir=cache_dir,
                                     train=True,
                                     split=1.)
surface_set = shapenet.ShapeNet_Surface_Meshes(root=shapenet_root,
                                               categories=categories,
                                               cache_dir=cache_dir,
                                               train=True,
                                               split=1.)

# %%
EPS = 1e-7
m = 4
n = 6
batch = 10
learning_rate = .01
iters = 500
dim = 3
sample_idx = 0
train_theta_sample_num = 30
points_sample_num = 3000
train_grid_sample_num = 5000

device_type = 'cuda:7'
#device_type = 'cpu'
train_periodic_after_abstraction = True

periodicnet = train_periodic_after_abstraction
device = torch.device(device_type)

if periodicnet:
    ocoef = 1.
    ccoef = 10.
else:
    ocoef = 1.
    ccoef = 1.

overlap_reg_coef = 1.
self_overlap_reg_coef = 1.

# points_num, dim
points = point_set[sample_idx]['data']['points'].to(device) * 10
all_points_sample_num = points.shape[0]

# grid_points_num, dim
xyz = sdf_set[sample_idx]['data']['sdf_points'].to(device) * 10
x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]
mesh = surface_set[sample_idx]['data']
meshkal = rep.TriangleMesh.from_tensors(mesh['vertices'] * 10, mesh['faces'])
meshkal.to('cuda:0')
sdf_func = kal.conversions.trianglemesh_to_sdf(meshkal, x.shape[0])
sgn = (sdf_func(xyz.to('cuda:0')).to(device) <= 0.001).float()
all_grid_sample_num = sgn.shape[0]


def get_target_sample():
    index = []
    for _ in range(batch):
        index_single = random.sample(range(all_grid_sample_num),
                                     train_grid_sample_num)
        index.extend(index_single)
    train_x = x[index]
    train_y = y[index]
    train_z = z[index]

    target_coord = torch.stack([train_x, train_y, train_z],
                               axis=1).view(batch, -1, dim)

    target_sgn = sgn[index].float().view(batch, -1)

    points_index = []
    for _ in range(batch):
        index_single = random.sample(range(all_points_sample_num),
                                     points_sample_num)
        points_index.extend(index_single)
    target_points = points[points_index, :].view(batch, -1, dim)

    return target_points, target_coord, target_sgn
