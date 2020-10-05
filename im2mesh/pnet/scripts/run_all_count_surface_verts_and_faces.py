import subprocess
config_paths = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/generation_primitive_wise_watertight_20200507_023808/gen_config_primitive_wise_watertight_20200507_023808.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/generation_explicit_20200502_003123/gen_config__20200502_003123.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn20_target_n_4096_no_overlap_reg2_20200502_043525/generation_explicit__20200502_144542/gen_config__20200502_144542.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn15_target_n_4096_no_overlap_reg_20200502_041907/generation_explicit__20200502_144453/gen_config__20200502_144453.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/generation_explicit__20200502_043815/gen_config__20200502_043815.yaml'
]
for path in config_paths:
    subprocess.run(
        'python3 /home/mil/kawana/workspace/occupancy_networks/im2mesh/pnet/scripts/count_surface_verts_and_faces.py {}'
        .format(path),
        shell=True,
        check=True)
