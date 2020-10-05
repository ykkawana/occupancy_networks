#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--explicit \
--unique_name f60k \
--data.is_generate_mesh true \
--data.patch_side_length 25
#--data.classes [\"02691156\"] \
