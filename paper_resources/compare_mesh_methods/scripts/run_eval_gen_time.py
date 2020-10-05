import subprocess
gpu_id = 0
arg_strs = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up0_20200505_052830/gen_config_implicit_up0_20200505_052830.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up1_20200505_052940/gen_config_implicit_up1_20200505_052940.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_implicit_up2_20200505_053138/gen_config_implicit_up2_20200505_053138.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_0_20200509_155825/gen_config_subdiv_0_20200509_155825.yaml  --explicit --generation.is_skip_surface_mask_generation_time true  --generation.is_just_measuring_time true',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_1_20200509_155936/gen_config_subdiv_1_20200509_155936.yaml  --explicit --generation.is_skip_surface_mask_generation_time true  --generation.is_just_measuring_time true',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_2_20200509_160000/gen_config_subdiv_2_20200509_160000.yaml  --explicit --generation.is_skip_surface_mask_generation_time true  --generation.is_just_measuring_time true',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit_subdiv_3_20200509_160021/gen_config_subdiv_3_20200509_160021.yaml  --explicit --generation.is_skip_surface_mask_generation_time true  --generation.is_just_measuring_time true',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/generation_explicit__20200502_185812/gen_config__20200502_185812.yaml  --explicit --generation.is_skip_surface_mask_generation_time true  --generation.is_just_measuring_time true',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/gen_config_primitive_wise_watertight_20200507_023808.yaml --generation.is_just_measuring_time true',
]
arg_strs = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/onet_pretrained_20200502_013054/pretrained__20200502_013403/gen_config__20200502_013403.yaml'
]
script = '/home/mil/kawana/workspace/occupancy_networks/eval_gen_time.py'

command_template = 'CUDA_VISIBLE_DEVICES={gpu_id} python3 {script} {arg}'
for arg in arg_strs:
    command = command_template.format(gpu_id=gpu_id, script=script, arg=arg)
    subprocess.run(command, shell=True, check=True)
