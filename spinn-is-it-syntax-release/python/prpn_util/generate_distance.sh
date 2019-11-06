#!/bin/sh
PT3_ENV_DIR="path_to_python_env"
PRPN_NAME="prpn_experiment_name"
ALLNLI_DIR="path_to_all_nli_raw_data_dir"
MODEL_PATH="path_to_best_prpn_checkpoint_named_by"$PRPN_NAME
source $PT3_ENV_DIR/bin/activate &&
CUDA_VISIBLE_DEVICES=0 python generate_distance.py \
--data $ALLNLI_DIR \
--cuda \
--tied \
--hard \
--device 0 \
--save $MODEL_PATH \
--save_data_prefix "path_to_output_all_nli_dataset_w/_distance_named_by"$PRPN_NAME \
--punctuations