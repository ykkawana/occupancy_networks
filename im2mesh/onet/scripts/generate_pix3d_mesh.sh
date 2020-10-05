#!/bin/bash
config=$1
GPU=$2
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $config \
--data.test_split master \
--data.path data/Pix3D \
--data.voxels_file null \
--data.is_generate_mesh true \
--unique_name pix3d_no_fit_to_gt \
--generation.is_fit_to_gt_loc_scale false \
--data.img_folder class_agnostic_margin_224
