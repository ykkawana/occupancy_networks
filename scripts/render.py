# %%
import eval_utils
from datetime import datetime
import os
os.chdir('/home/mil/kawana/workspace/occupancy_networks')

path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef3_20200612_002143/generation_explicit__20200612_002246/meshes/04379243/ef4bc194041cb83759b7deb32c0c31a.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_no_regs_normal_by_radius_gradient_coef3_20200612_002143/generation_explicit__20200612_002316/meshes/04379243/ef4bc194041cb83759b7deb32c0c31a.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/generation__20200502_034223/meshes/03001627/dbfab57f9238e76799fc3b509229d3d.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/03001627/dbfab57f9238e76799fc3b509229d3d.off'

path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/generation__20200502_034223/meshes/03001627/cc70b9c8d4faf79e5a468146abbb198.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/03001627/cc70b9c8d4faf79e5a468146abbb198.off'
path = '/data/ugui0/kawana/ShapeNetCore.v1/03001627/cc70b9c8d4faf79e5a468146abbb198/model.obj'

path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit__20200502_043815/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn6_target_n_4096_no_overlap_reg_20200502_042856/generation_explicit__20200502_144128/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn3_target_n_4096_no_overlap_reg_20200502_042724/generation_explicit__20200502_144107/meshes/03001627/d2f844904a5cf31db93d537020ed867c.off'
path = '/data/ugui0/kawana/ShapeNetCore.v1/03001627/d2f844904a5cf31db93d537020ed867c/model.obj'
rendering_script_path = '/home/mil/kawana/workspace/occupancy_networks/scripts/render_3dobj.sh'
date_str = datetime.now().strftime(('%Y%m%d_%H%M%S'))
rendering_out_dir = os.path.join('./')
camera_param_path = os.path.join(rendering_out_dir, 'camera_param.txt')

eval_utils.render_by_blender(rendering_script_path,
                             camera_param_path,
                             path,
                             rendering_out_dir,
                             'mesh_rendering_{}'.format(date_str),
                             skip_reconvert=True,
                             use_cycles=True,
                             use_lamp=True)

# %%
