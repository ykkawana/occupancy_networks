
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD && export PYTHONPATH=$PYTHONPATH:external/periodic_shapes
CUDA_VISIBLE_DEVICES=$1 python3 train.py configs/img/plane_pnet_manual_hps_5.yaml

