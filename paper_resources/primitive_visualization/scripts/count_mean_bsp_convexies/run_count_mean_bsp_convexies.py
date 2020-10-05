
import subprocess

gpus = [0, 1, 2, 3, 6, 7]
script = '/home/mil/kawana/workspace/occupancy_networks/paper_resources/primitive_visualization/scripts/count_mean_bsp_convexies/count_mean_bsp_convexies.py'
config_paths = [
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn3_20200501_191343/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn6_20200501_191325/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn8_20200502_134305/eval_config_20200502_134305.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn10_20200501_191222/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn15_20200504_012215/eval_config_20200504_012215.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn20_20200502_133608/eval_config_20200502_133608.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn30_20200501_191145/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn50_20200504_013233/eval_config_20200504_013233.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn100_20200501_190916/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn150_20200501_190907/config.yaml',
    '/home/mil/kawana/workspace/occupancy_networks/out/submission/eval/img/bspnet_pn256_author_provided_20200501_184850/config.yaml'
]
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
