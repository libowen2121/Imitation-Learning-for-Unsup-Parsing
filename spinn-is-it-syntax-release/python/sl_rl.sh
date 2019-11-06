#!/bin/sh
# learning from demonstrations + reinforcement learning
PT3_ENV_DIR="path_to_python_env"
PRPN="prpn_experiment_name"
GLOVE="path_to_glove"
ALLNLI_EVAL_DIST="path_to_allnli_valid_data_w/_distances"                   # test set
ALLNLI_TRAIN_TRAIN_DIST="path_to_pregenerated_training_data_w/_distances"   # trainig set
ALLNLI_TRAIN_VALID_DIST="path_to_pregenerated_valid_data_w/_distances"      # dev set
RESULT="path_to_result_dir" 
source $PT3_ENV_DIR/bin/activate &&
CUDA_VISIBLE_DEVICES=0 python -m spinn.models.supervised_classifier \
--prpn_name $PRPN \
--experiment_name "XXX"$PRPN \
--nocomposition_ln \
--pyramid_trainable_temperature  \
--eval_seq_length 999 \
--seq_length 80 \
--pyramid_temperature_decay_per_10k_steps 1.0 \
--data_type nli \
--log_path $RESULT \
--embedding_keep_rate 1.0 \
--training_data_path $ALLNLI_TRAIN_TRAIN_DIST \
--batch_size 32 \
--mlp_dim 1024 \
--encode_bidirectional  \
--statistics_interval_steps 100 \
--embedding_data_path $GLOVE \
--semantic_classifier_keep_rate 0.70 \
--model_type ChoiPyramid \
--model_dim 600 \
--num_mlp_layers 1 \
--learning_rate_decay_per_10k_steps 1.0 \
--word_embedding_dim 300 \
--encode gru \
--ckpt_path $RESULT \
--sample_interval_steps 1000 \
--eval_data_path $ALLNLI_TRAIN_VALID_DIST:$ALLNLI_EVAL_DIST \
--gpu 0 \
--tree_joint \
--distance_type definition \
--learning_rate 3.0e-5 \
--l2_lambda 1.0e-5 \
--sbs_weight 3.0e-2 \
--sbs_step 200000 \
--eval_interval_steps 1000 \
--early_stopping_steps_to_wait 100000
# --save_sl \
# --save_sl_step 1000