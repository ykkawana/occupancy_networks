#!/bin/bash
config=$1
GPU=${2:-0}
UP=${3:-2}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--generation.upsampling_steps $UP \
--unique_name implicit_up$UP
#--unique_name max_surface_extract \
#--model.decoder_kwargs.extract_surface_point_by_max true
#--data.classes [\"02691156\"] \
