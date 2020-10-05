gpu=$2

CUDA_VISIBLE_DEVICES=$gpu python3 train.py /home/mil/kawana/workspace/occupancy_networks/configs/img/atlasnetv2_a2s_a_pn$1.yaml --use_written_out_dir
CUDA_VISIBLE_DEVICES=$gpu python3 train.py /home/mil/kawana/workspace/occupancy_networks/configs/img/atlasnetv2_a2s_s_pn$1.yaml --use_written_out_dir
