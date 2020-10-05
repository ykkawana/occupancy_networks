#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--data.voxels_file null \
--generation.is_gen_primitive_wise_watertight_mesh true \
--data.bspnet.path data/Pix3DBSP \
--data.bspnet.img_folder bsp_class_agnostic_margin_224 \
--data.test_split master \
--data.path data/Pix3D \
--unique_name pix3d_debug \
--data.img_folder class_agnostic_margin_224 \
--data.bspnet.force_disable_bspnet_mode true \
--generation.is_fit_to_gt_loc_scale true \
--generation.bspnet.is_skip_realign true
