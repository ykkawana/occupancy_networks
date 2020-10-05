# %%
import trimesh
import os
os.chdir('/home/mil/kawana/workspace/occupancy_networks/disposable')
from im2mesh.utils.libmesh import check_mesh_contains
import numpy as np

# %%
mesh_for_iou = trimesh.load(
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/meshes/03691459/c9de3e18847044da47e2162b6089a53e.off'
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/meshes/02691156/d1cdd239dcbfd018bbf3143b1cb6076a.off'
)
meshes = mesh_for_iou.split()
if len(meshes) != 0:
    for idx, mesh in enumerate(meshes):
        if idx == 0:
            occ = check_mesh_contains(mesh, mesh_for_iou.vertices)
        else:
            occ |= check_mesh_contains(mesh, mesh_for_iou.vertices)

vis = np.logical_not(occ)
v1 = np.take(vis, mesh_for_iou.faces[:, 0])
v2 = np.take(vis, mesh_for_iou.faces[:, 1])
v3 = np.take(vis, mesh_for_iou.faces[:, 2])

v = v1 | v2 | v3

index = np.take(np.arange(len(v)), np.nonzero(v)[0])
mesh_for_iou.faces = mesh_for_iou.faces[index, :]

mesh_for_iou.show()

# %%
mesh_normals = mesh_for_iou.face_normals
total_surface_verts_count = 0
sampled_verts = []
sampled_normals = []
k100 = 100000
import plotly.graph_objects as go
while True:
    verts, idx = mesh_for_iou.sample(3 * k100, return_index=True)
    normals = mesh_normals[idx, :]
    surface_verts = verts + normals * 1e-4

    for idx, mesh in enumerate(meshes):
        if idx == 0:
            occ = check_mesh_contains(mesh, surface_verts)
        else:
            occ |= check_mesh_contains(mesh, surface_verts)

    surface_idx = np.logical_not(occ)
    total_surface_verts_count += surface_idx.sum()
    sampled_verts.append(verts[surface_idx, :])
    sampled_normals.append(normals[surface_idx, :])
    if total_surface_verts_count > k100:
        break

surface_normals = np.concatenate(sampled_normals, axis=0)
surface_verts = np.concatenate(sampled_verts, axis=0)

# %%
surface_verts = np.load(
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/meshes/02691156/d1b28579fde95f19e1873a3963e0d14_vertex_attributes_surface_sample100k.npz'
)['vertices']
select_idx = np.random.choice(np.arange(len(surface_verts)),
                              3000,
                              replace=False)
surface_verts = surface_verts[select_idx, :]

marker_opt = dict(size=1)
plots = []
x1 = surface_verts[:, 0]
y1 = surface_verts[:, 1]
z1 = surface_verts[:, 2]
plots.append(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=marker_opt))

fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
fig.show()

# %%
import pickle
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/surface_verts_faces_count.pkl'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn20_target_n_4096_no_overlap_reg2_20200502_043525/generation_explicit__20200502_144542/surface_verts_faces_count.pkl'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn15_target_n_4096_no_overlap_reg_20200502_041907/generation_explicit__20200502_144453/surface_verts_faces_count.pkl'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit__20200502_043815/surface_verts_faces_count.pkl'
df = pickle.load(open(path, 'rb')).mean()
coeff = df['faces_out'] / df['faces_num']
p = 50
s = 28
len(
    trimesh.creation.uv_sphere(theta=np.linspace(0, 1, s),
                               phi=np.linspace(0, 1,
                                               s)).faces) * p * coeff.item()
