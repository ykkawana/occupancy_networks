import subprocess
import os

gpus = [7, 6, 5, 4]

config_paths = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn3_20200503_220859/eval_config_20200503_220859.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn6_20200503_220859/eval_config_20200503_220859.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn8_20200503_222342/eval_config_20200503_222342.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn10_20200503_220859/eval_config_20200503_220859.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn15_20200503_221008/eval_config_20200503_221008.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn20_20200503_220954/eval_config_20200503_220954.yaml',
    #'/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn30_20200503_221032/eval_config_20200503_221032.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn50_20200503_221053/eval_config_20200503_221053.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn3_20200501_191343/gen_part_label_20200504_210612.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn6_20200501_191325/gen_part_label_20200504_210301.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn8_20200502_134305/gen_part_label_20200504_210440.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/gen_part_label_20200504_210456.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn15_20200504_012215/gen_part_label_20200504_210551.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn20_20200502_133608/gen_part_label_20200504_210904.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn50_20200504_013233/gen_part_label_20200504_210240.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn3_target_n_4096_no_overlap_reg_20200502_042724/eval_config_20200502_161510.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn6_target_n_4096_no_overlap_reg_20200502_042856/eval_config_20200502_161710.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn8_target_n_4096_no_overlap_reg_20200502_042147/eval_config_20200502_161812.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/eval_config_20200502_105908.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn15_target_n_4096_no_overlap_reg_20200502_041907/eval_config_20200502_162038.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn20_target_n_4096_no_overlap_reg2_20200502_043525/eval_config_20200502_162127.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_20200502_104902/eval_config_20200502_173040.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/eval_config_20200502_172135.yaml',
    #     '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/eval_config_20200502_200733.yaml'
]
script = 'generate_part_label.py'

config_paths = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn3_20200501_191343/part_assignment_20200505_025509.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn6_20200501_191325/part_assignment_20200505_025549.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn8_20200502_134305/part_assignment_20200505_025607.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/part_assignment_20200505_025630.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn15_20200504_012215/part_assignment_20200505_025652.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn20_20200502_133608/part_assignment_20200505_025712.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn50_20200504_013233/part_assignment_20200505_025731.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn3_target_n_4096_no_overlap_reg_20200502_042724/part_assignment_20200505_025835.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn6_target_n_4096_no_overlap_reg_20200502_042856/part_assignment_20200505_025848.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn8_target_n_4096_no_overlap_reg_20200502_042147/part_assignment_20200505_025907.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn10_target_n_4096_no_overlap_reg_20200502_001739/part_assignment_20200505_025922.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn15_target_n_4096_no_overlap_reg_20200502_041907/part_assignment_20200505_025938.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn20_target_n_4096_no_overlap_reg2_20200502_043525/part_assignment_20200505_025956.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_20200502_104902/part_assignment_20200505_030037.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn30_no_regs_no_normal_20200502_202018/part_assignment_20200505_030058.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/pnet_finetue_only_transition_cceff10_pn50_target_n_4096_no_overlap_reg_20200502_041227/part_assignment_20200505_030119.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn3_20200503_220859/part_assignment_20200505_034110.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn6_20200503_220859/part_assignment_20200505_034110.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn8_20200503_222342/part_assignment_20200505_034108.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn10_20200503_220859/part_assignment_20200505_034110.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn15_20200503_221008/part_assignment_20200505_034110.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn20_20200503_220954/part_assignment_20200505_034110.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/atlasnetv2_pn50_20200503_221053/part_assignment_20200505_034110.yaml'
]
script = 'eval_part_label.py'

le = int(len(config_paths) // 4) + 1
procs = []

for ii in range(le):
    for jj in range(len(gpus)):
        idx = len(gpus) * ii + jj
        if idx > len(config_paths) - 1:
            break
        command = 'CUDA_VISIBLE_DEVICES={gpu} python3 {script} {config}'.format(
            gpu=gpus[jj], config=config_paths[idx], script=script)
        print(command)
        proc = subprocess.Popen(command, shell=True)
        procs.append(proc)

for proc in procs:
    proc.communicate()
