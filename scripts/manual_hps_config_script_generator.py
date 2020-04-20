import yaml
import uuid
import os

original_path = 'configs/img/plane_pnet_manual_hps.yaml'
dirpath = 'manual_hps_loss_weights_scripts'
script_template = """
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD && export PYTHONPATH=$PYTHONPATH:external/periodic_shapes
CUDA_VISIBLE_DEVICES=$1 python3 train.py {configpath}
"""

if not os.path.exists(dirpath):
    os.mkdir(dirpath)

occupancy_loss_coef_range = [0, 0.1, 1, 10]
chamfer_loss_coef_range = [0, 0.1, 1, 10]

params = yaml.load(open(original_path))
cnt = 0
port = 8899
for ow in occupancy_loss_coef_range:
    for cw in chamfer_loss_coef_range:
        if ow == cw:
            continue
        copied = params.copy()
        copied['trainer']['occupancy_loss_coef'] = ow
        copied['trainer']['chamfer_loss_coef'] = cw
        copied['training']['out_dir'] = os.path.join('out/img',
                                                     'pnet_hps_{}'.format(cnt))

        outfilepath = original_path.split('.')[0] + '_{}.yaml'.format(cnt)
        yaml.dump(copied, open(outfilepath, 'w'))

        scriptpath = os.path.join(dirpath, 'hps_{}.sh'.format(cnt))
        script_content = script_template.format(configpath=outfilepath,
                                                port=(port + cnt))
        with open(scriptpath, 'w') as f:
            print(script_content, file=f)
        cnt += 1
