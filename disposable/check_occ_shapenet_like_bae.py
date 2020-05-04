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
occ_shapenet_base_dir = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNet'
bae_model_path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetBAESemSeg/02691156/1a04e3eab45ca15dd86060f189eb133/bae_semseg_labeled_pointcloud.npz'

class_id = bae_model_path.split('/')[-3]
modelname = bae_model_path.split('/')[-2]
occ_model_path = os.path.join(occ_shapenet_base_dir, class_id, modelname,
                              'pointcloud.npz')

occ_points = np.load(occ_model_path)['points']
bae_points = np.load(bae_model_path)['points']

plots = []
marker_opt = dict(size=1)

select = random.sample(range(len(occ_points)), 100)
x1 = occ_points[select, 0]
y1 = occ_points[select, 1]
z1 = occ_points[select, 2]
plots.append(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=marker_opt))

select = random.sample(range(len(bae_points)), 100)
x2 = bae_points[select, 0]
y2 = bae_points[select, 1]
z2 = bae_points[select, 2]
plots.append(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=marker_opt))

fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()
