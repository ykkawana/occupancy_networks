#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--explicit \
--data.is_generate_mesh true
#--data.classes [\"02691156\"] \
