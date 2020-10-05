#!/bin/bash
config=$1
GPU=$2
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $config \
--explicit \
--data.test_split master \
--data.icosahedron_subdiv 4 \
--data.path data/Pix3D \
--data.voxels_file null \
--data.img_folder class_agnostic_margin_224 \
--test.is_eval_explicit_mesh true \
--generation.is_explicit_mesh true \
--unique_name pix3d_no_fit_to_gt \
--generation.is_fit_to_gt_loc_scale false \
--data.is_normal_icosahedron true

