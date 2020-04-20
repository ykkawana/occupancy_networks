# %%
from torch_geometric.nn.conv import MessagePassing
import torch
from torch_geometric.data import Data
import trimesh



# %%
mesh = trimesh.load('/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_20200413_015954/generation_explicit/vis/03001627_chair/00_mesh.off')
mesh.show()
print(mesh.vertices.shape)
#data = Data(x=x, edge_index=edge_index)
