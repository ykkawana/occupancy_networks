import subprocess
import os

gpus = [7, 6, 5, 4]

config_paths = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_f60k_20200511_190709/gen_config_f60k_20200511_190709.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_cceff0_20200502_202829/generation_explicit_f60k_20200511_191434/gen_config_f60k_20200511_191434.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_oceff0_20200502_203001/generation_explicit_f60k_20200511_191411/gen_config_f60k_20200511_191411.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_oceff0_no_surface_loss_20200502_202538/generation_explicit_f60k_20200511_191447/gen_config_f60k_20200511_191447.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn15_target_n_4096_no_overlap_reg_20200502_041907/generation_explicit_f60k_20200511_190844/gen_config_f60k_20200511_190844.yaml',
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn20_target_n_4096_no_overlap_reg2_20200502_043525/generation_explicit_f60k_20200511_190903/gen_config_f60k_20200511_190903.yaml',
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_f60k_20200511_163104/gen_config_f60k_20200511_163104.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_0_20200509_155825/gen_config_subdiv_0_20200509_155825.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_2_20200509_160000/gen_config_subdiv_2_20200509_160000.yaml',
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit_f60k_20200511_190925/gen_config_f60k_20200511_190925.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/generation_split_watertight_20200511_170248/gen_config_split_watertight_20200511_170248.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn15_20200504_012215/generation_split_watertight_20200511_170245/gen_config_split_watertight_20200511_170245.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn20_20200502_133608/generation_split_watertight_20200511_170255/gen_config_split_watertight_20200511_170255.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn30_20200501_191145/generation_split_watertight_20200511_170649/gen_config_split_watertight_20200511_170649.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn50_20200504_013233/generation_split_watertight_20200511_170259/gen_config_split_watertight_20200511_170259.yaml',
]
script = '/home/mil/kawana/workspace/occupancy_networks/eval_fscore_cd1_from_vertex_attribute_surface_100k.py'

le = int(len(config_paths) // 4) + 1
procs = []

for ii in range(le):
    for jj in range(len(gpus)):
        idx = len(gpus) * ii + jj
        if idx > len(config_paths) - 1:
            break
            #command = 'CUDA_VISIBLE_DEVICES={gpu} python3 {script} {config}'.format(
        command = 'python3 {script} {config} {gpu}'.format(
            gpu=gpus[jj], config=config_paths[idx], script=script)
        print(command)
        proc = subprocess.Popen(command, shell=True)
        procs.append(proc)

for proc in procs:
    proc.communicate()
