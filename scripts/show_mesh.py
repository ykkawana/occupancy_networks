# %%
import trimesh
import plotly.graph_objects as go
import numpy as np
# %%

path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_normal_transition_feature_concat_color_feature_20200425_154337/generation_explicit_20200425_154727/vis/03636649_lamp/00_mesh.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_normal_transition_feature_concat_color_feature_20200425_154337/generation_explicit_20200425_154727/meshes/03636649/d1b15263933da857784a45ea6efa1d77.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_target_n_4096_no_overlap_reg_normal_transition_feature_concat_color_feature_20200425_154337/generation_explicit_20200425_154727/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
path = '/home/mil/kawana/workspace/occupancy_networks/external/bspnet/samples/bsp_svr_out/0_bsp.obj'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/meshes/02691156/f59a2be8fd084418bbf3143b1cb6076a.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_whole_mesh_20200602_210619/meshes/03636649/d97a86cea650ae0baf5b49ad7809302.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_whole_mesh_20200602_211723/meshes/03636649/f2f6fbeacda7cbecfcb8d8c6d4df8143.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef1_20200612_002121/generation_explicit__20200612_002246/meshes/03636649/d1b15263933da857784a45ea6efa1d77.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_finetune_no_reg_normal_coef1_20200612_002207/generation_explicit__20200612_002356/meshes/03636649/d1b15263933da857784a45ea6efa1d77.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef1_20200612_002121/generation_explicit__20200612_002246/meshes/02691156/d1b407350e61150942d79310bc7e47b3.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef3_20200612_002143/generation_explicit__20200612_002316/meshes/02691156/d1b1c13fdec4d69ccfd264a25791a5e1.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef3_20200612_002143/generation_explicit__20200612_002316/meshes/02691156/d1b407350e61150942d79310bc7e47b3.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_finetune_no_reg_normal_coef1_20200612_002207/generation_explicit__20200612_002356/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/02691156/d1a8e79eebf4a0b1579c3d4943e463ef.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/03636649/d1aed86c38d9ea6761462fc0fa9b0bb4.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef1_20200612_002121/generation_explicit__20200612_002246/meshes/02691156/d2e2e23f5be557e2d1ab3b031c100cb1.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef3_20200612_002143/generation_explicit__20200612_002316/meshes/02691156/d1cdd239dcbfd018bbf3143b1cb6076a.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/pointcloud/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_holes_overfit_chair_split_one2_20200811_234832/generation_explicit__20200811_234939/meshes/03001627/d2c465e85d2e8f1fcea003eff0268278.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/pointcloud/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_holes_overfit_chair_deskchair_20200811_235251/generation_explicit__20200811_235402/meshes/03001627/d1ec6e9b8063b7efd7f7a4c4609b0913.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/pointcloud/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_holes_overfit_chair_split_one2_moresamples_20200812_192828/generation_explicit__20200812_192956/meshes/03001627/d2c465e85d2e8f1fcea003eff0268278.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/pointcloud/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_holes_overfit_chair_split_one2_moresamples_20200812_192828/generation_explicit__20200812_194404/meshes/03001627/d2c465e85d2e8f1fcea003eff0268278.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit__20200502_043815/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_implicit_up2_20200506_053726/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn6_target_n_4096_no_overlap_reg_20200502_042856/generation_explicit__20200502_144128/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/generation__20200502_034223/meshes/03001627/dbfab57f9238e76799fc3b509229d3d.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/generation__20200502_034223/meshes/03001627/cc70b9c8d4faf79e5a468146abbb198.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/03001627/cc70b9c8d4faf79e5a468146abbb198.off'
mesh = trimesh.load(path)

mesh.show()

# %%
"""
path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetAtlasNetV2Converted/02691156/83778fc8ddde4a937d5bc8e1d7e1af86/pointcloud.npz'
path = '/home/mil/kawana/workspace/occupancy_networks/data/ShapeNetAtlasNetV2Converted/04256520/1bce3a3061de03251009233434be6ec0/pointcloud.npz'

points = np.load(path)['points']
path = path.replace('ShapeNetAtlasNetV2Converted', 'ShapeNet')
points2 = np.load(path)['points']

marker_opt = dict(size=1)
plots = []

select = np.random.choice(np.arange(len(points)), size=3000, replace=False)
points = points[select, :]
x1 = points[:, 0]
y1 = points[:, 1]
z1 = points[:, 2]
plots.append(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=marker_opt))

select = np.random.choice(np.arange(len(points2)), size=3000, replace=False)
points2 = points2[select, :]
x2 = points2[:, 0]
y2 = points2[:, 1]
z2 = points2[:, 2]
plots.append(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=marker_opt))
plots.append(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=marker_opt))

fig = go.Figure(data=plots)
fig.update_layout(scene_aspectmode='data')
#fig.show()

"""
