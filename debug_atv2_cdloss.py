# %%
import sys
sys.path.insert(
    0, '/home/mil/kawana/workspace/occupancy_networks/external/atlasnetv2')
sys.path.insert(
    0,
    '/home/mil/kawana/workspace/occupancy_networks/external/periodic_shapes')
sys.path.insert(0, '/home/mil/kawana/workspace/occupancy_networks')
from atlasnetv2.extension import dist_chamfer
from im2mesh.atlasnetv2 import utils as atv2_utils
import torch
distChamferL2 = dist_chamfer.chamferDist()

# %%
source = torch.rand([3, 100, 3]).float().to('cuda') * 2
target = torch.rand([3, 200, 3]).float().to('cuda') * 5
dist1, dist2 = distChamferL2(source, target)
loss1 = dist1.mean() + dist2.mean()
loss2 = atv2_utils.chamfer_loss(source, target, 'l2')
print(loss1.item(), loss2.item())

# %%
