#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--explicit \
--data.voxel_file null \
--data.is_normal_icosahedron true \
--data.icosahedron_subdiv 4 \
--test.is_eval_explicit_mesh true \
--generation.is_explicit_mesh true
#--unique_name max_surface_extract \
#--model.decoder_kwargs.extract_surface_point_by_max true
#--data.classes [\"02691156\"] \
