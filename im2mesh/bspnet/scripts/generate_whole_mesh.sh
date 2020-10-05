#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--unique_name whole_mesh \
--generation.is_gen_whole_mesh true \
--generation.is_gen_skip_vertex_attributes true
#--data.classes [\"02691156\"] \
