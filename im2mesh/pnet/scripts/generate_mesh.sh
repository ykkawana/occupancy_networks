#!/bin/bash
config=$1
GPU=${2:-0}
length=$3
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--explicit \
--data.voxel_file null \
--data.is_normal_icosahedron true \
--data.icosahedron_subdiv 4 \
--test.is_eval_explicit_mesh true \
--generation.is_explicit_mesh true
# Gen icosahedron mesh

# Gen controled mesh
--explicit \
--unique_name f60k \
--data.voxel_file null \
--data.is_normal_uv_sphere true \
--data.uv_sphere_length $length \
--test.is_eval_explicit_mesh true \
--generation.is_explicit_mesh true

