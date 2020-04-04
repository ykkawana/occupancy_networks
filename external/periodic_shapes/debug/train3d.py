import sys
sys.path.insert(0, '.')
import kaolin as kal
from kaolin.datasets import shapenet
from kaolin import rep
from kaolin import conversions
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
from models import periodic_shape_sampler_xyz
from models import super_shape_sampler
from models import super_shape
from models import model_utils
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
from visualize import plot


seed = 0
random.seed(seed)  
np.random.seed(seed)  
# PyTorch のRNGを初期化  
torch.manual_seed(seed)


dotenv.load_dotenv(verbose=True)

category = 'plane'
cache_root = os.getenv('SHAPENET_KAOLIN_CACHE_ROOT')
shapenet_root = os.getenv('SHAPENET_ROOT')
cache_dir = os.path.join(cache_root, category)

categories = [category]


sdf_set = shapenet.ShapeNet_SDF_Points(root=shapenet_root, categories=categories, cache_dir=cache_dir, train=True, split=1.)
point_set = shapenet.ShapeNet_Points(root=shapenet_root, categories=categories, cache_dir=cache_dir, train=True, split=1.)
surface_set = shapenet.ShapeNet_Surface_Meshes(root=shapenet_root, categories=categories, cache_dir=cache_dir, train=True, split=1.)


EPS = 1e-7
m = 4
n = 6
batch = 10
learning_rate = .01
iters = 10
dim = 3
sample_idx = 0
train_theta_sample_num = 10
points_sample_num = 2000
train_grid_sample_num = 3000

device_type = 'cuda:7'
#device_type = 'cpu'
train_periodic_after_abstraction = False

periodicnet = train_periodic_after_abstraction
device = torch.device(device_type)

if periodicnet:
    ocoef = 1.
    ccoef = 10.
else:
    ocoef = 1.
    ccoef = 1.

overlap_reg_coef = 1.

# points_num, dim
points = point_set[sample_idx]['data']['points'].to(device)
all_points_sample_num = points.shape[0]

# grid_points_num, dim
xyz = sdf_set[sample_idx]['data']['sdf_points'].to(device)
x = xyz[:, 0]
y = xyz[:, 0]
z = xyz[:, 0]
mesh = surface_set[sample_idx]['data']
meshkal = rep.TriangleMesh.from_tensors(mesh['vertices'],
                                    mesh['faces'])
meshkal.to('cuda:0')
sdf_func = kal.conversions.trianglemesh_to_sdf(meshkal, x.shape[0])
sgn = (sdf_func(xyz.to('cuda:0')).to(device) <= 0.001).float()
all_grid_sample_num = sgn.shape[0]

def get_target_sample():
    index = []
    for _ in range(batch):
        index_single = random.sample(range(all_grid_sample_num), train_grid_sample_num)
        index.extend(index_single)
    train_x = x[index]
    train_y = y[index]
    train_z = z[index]

    target_coord = torch.stack([train_x, train_y, train_z], axis=1).view(batch, -1, dim)

    target_sgn = sgn[index].float().view(batch, -1)

    points_index = []
    for _ in range(batch):
        index_single = random.sample(range(all_points_sample_num), points_sample_num)
        points_index.extend(index_single)
    target_points = points[points_index, :].view(batch, -1, dim)

    return target_points, target_coord, target_sgn


if train_periodic_after_abstraction:
    #primitive.eval()
    pass
else:
    primitive = super_shape.SuperShapes(m, n, quadrics=True, train_ab=False, dim=dim)
    primitive.to(device)


if periodicnet or train_periodic_after_abstraction:
    print('Train periodic net')
    sampler = periodic_shape_sampler_xyz.PeriodicShapeSamplerXYZ(points_sample_num, m, n, factor=2, dim=dim)
    optimizer = optim.Adam([*sampler.parameters(), *primitive.parameters()], lr=learning_rate)
else:
    print('Train super shape')
    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    optimizer = optim.Adam(primitive.parameters(), lr=learning_rate)

sampler.to(device)
torch.autograd.set_detect_anomaly(True)

loss_log = []
for idx in tqdm.tqdm(range(iters)):
    optimizer.zero_grad()

    # Ensure polar coordinate samples are closed at 0 and 2 pi.
    thetas = utils.sample_spherical_angles(batch=batch, sample_num=train_theta_sample_num, sampling='uniform', device=device, dim=dim)

    target_points, target_coord, target_sgn = get_target_sample()

    kwargs = {
        'thetas': thetas,
        'coord': target_coord
    }
    if periodicnet:
        kwargs['points'] = target_points

    param = primitive()
    #print(param['m_vector'].device)
    prd_points, prd_mask, prd_tsd = sampler(param, **kwargs)
    prd_sgn = model_utils.convert_tsd_range_to_zero_to_one(prd_tsd).sum(1)

    overlap_reg = (nn.functional.relu(prd_sgn - 1.2).abs()).mean()

    oloss = nn.functional.binary_cross_entropy_with_logits(prd_sgn.clamp(min=1e-7), target_sgn)

    closs = custom_chamfer_loss.custom_chamfer_loss(prd_points, target_points, surface_mask=prd_mask, prob=None)

    reg = overlap_reg * overlap_reg_coef
    loss = closs * ccoef + oloss * ocoef + reg

    print(closs.item(), reg.item(), loss.item(), oloss.item(), reg.item())
    loss_log.append(closs.detach().cpu().numpy())
    loss.backward()
    optimizer.step()

primitive.eval()
sampler.eval()


from visualize import plot
points_list = [prd_points]
fig = plt.figure()
for idx, points in enumerate(points_list):
    plot.plot_primitive_point_cloud_3d(points)

surface_points1 = sampler.extract_super_shapes_surface_point(prd_points[0, ...].unsqueeze(0), primitive(), points=target_points[0, ...].unsqueeze(0))
surafce_points_list = [surface_points1]
fig = plt.figure()
for idx, surface_points in enumerate(surafce_points_list):
    plot.plot_primitive_point_cloud_3d(surface_points)

tsd_list = [prd_tsd]
fig = plt.figure()
for idx, tsd in enumerate(tsd_list):
    plot.draw_primitive_inside_2d(tsd, target_coord)





