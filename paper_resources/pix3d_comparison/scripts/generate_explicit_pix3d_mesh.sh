#!/bin/bash
config=$1
GPU=$2
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $config \
--explicit \
--data.test_split master \
--data.icosahedron_subdiv 4 \
--data.path data/Pix3D \
--generation.generation_dir generation_pix3d_class_agnostic_margin_224 \
--unique_name pix3d_class_agnostic_margin_224 \
--data.img_folder class_agnostic_margin_224 \
--data.is_normal_icosahedron true

