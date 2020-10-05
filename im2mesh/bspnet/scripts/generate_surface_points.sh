#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--unique_name surface_points \
--generation.is_gen_surface_points true \
--test.vertex_attribute_filename vertex_attributes_surface_sample100k
#--data.classes [\"02691156\"] \
