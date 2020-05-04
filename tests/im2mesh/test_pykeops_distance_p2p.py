# %%
import os
os.chdir('/home/mil/kawana/workspace/occupancy_networks')
from im2mesh import eval
import torch
import numpy as np
from im2mesh.utils.libkdtree import KDTree

# %%
source = np.random.rand(5, 3) * 10
target = np.random.rand(7, 3) * 10

kdtree = KDTree(source)
dist, idx = kdtree.query(target)

dist2, idx2 = eval.one_sided_chamfer_distance_with_index(target, source)
dist2_numpy = dist2.to('cpu').numpy()
idx2_numpy = idx2.to('cpu').numpy()

assert np.allclose(dist2_numpy, dist), (dist2_numpy, dist)
assert np.all(idx2_numpy == idx), (idx2_numpy, idx)

# %%
source_normals = np.random.rand(*source.shape)
target_normals = np.random.rand(*target.shape)

dist, normal_cost = eval.distance_p2p(source,
                                      source_normals,
                                      target,
                                      target_normals,
                                      mode='original')

dist2, normal_cost2 = eval.distance_p2p(source,
                                        source_normals,
                                        target,
                                        target_normals,
                                        mode='pykeops')

assert np.allclose(dist, dist2)
assert np.allclose(normal_cost, normal_cost2)
# %%
