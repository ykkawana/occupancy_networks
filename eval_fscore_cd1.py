import subprocess
import sys
import time

config_path = sys.argv[1]
gpu_id = sys.argv[2]
skip_cd1 = False
command_cd1 = 'CUDA_VISIBLE_DEVICES={gpu_id} python3 eval_meshes.py {config_path} {args}'
command_fsc = 'CUDA_VISIBLE_DEVICES={gpu_id} python3 eval_fscore.py {config_path} {args}'
procs = []
command3 = command_cd1.format(
    gpu_id=gpu_id,
    config_path=config_path,
    args=
    '--unique_name mesh_iou_normalize --test.is_normalize_by_side_length true --test.eval_from_vertex_attributes true'
    #'--unique_name mesh_iou_normalize --test.is_normalize_by_side_length true'
    #'--unique_name mesh_iou_normalize --test.is_normalize_by_side_length true --test.is_eval_iou_by_split true'
)
print(command3)
command4 = command_fsc.format(
    gpu_id=gpu_id,
    config_path=config_path,
    args=
    '--unique_name mesh_iou_normalize --test.is_normalize_by_side_length true  --test.eval_from_vertex_attributes true'
    #'--unique_name mesh_iou_normalize --test.is_normalize_by_side_length true'
    #'--unique_name mesh_iou_normalize_sample100k --test.is_normalize_by_side_length true --test.eval_from_vertex_attributes true --test.vertex_attribute_filename vertex_attributes_sample100k'
)
print(command4)
if not skip_cd1:
    proc_cd1 = subprocess.Popen(command3, shell=True)
    procs.append(proc_cd1)
    time.sleep(60)
proc_fsc = subprocess.Popen(command4, shell=True)
procs.append(proc_fsc)

for proc in procs:
    proc.communicate()
