#!/bin/bash
config=$1
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python3 generate.py $1 \
--unique_name implicit \
--generation.is_gen_implicit_mesh true
