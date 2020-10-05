#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--unique_name debug_primitive_wise_split \
--generation.is_gen_primitive_wise_watertight_mesh_debugged true \
--data.classes [\"02691156\"]
