#!/bin/bash

param=$1
model_path=$2
out=$3
name=$4
skip_reconvert_flag=${5:-"false"}
RENDER_FOR_CNN_PATH=/home/mil/kawana/workspace/RenderForCNN

PYTHONPATH=$PYTHONPATH:RENDER_FOR_CNN_PATH \
/home/mil/kawana/workspace/blender-2.79-linux-glibc219-x86_64/blender \
${RENDER_FOR_CNN_PATH}/render_pipeline/blank.blend \
--background \
--python ${RENDER_FOR_CNN_PATH}/render_pipeline/render_model_views.py  \
${skip_reconvert_flag} \
${model_path} \
${name}  ${param} ${out}
