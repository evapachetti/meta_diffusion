#!/bin/bash

# Example script to run the model

# Define paths for CSV files
CSV_PATH=''
TRAIN_PATH=${CSV_PATH}/meta_train.csv
VAL_PATH=${CSV_PATH}/meta_val.csv
TEST_PATH=${CSV_PATH}/meta_test.csv

# Remove carriage return characters from paths
TRAIN_PATH="${TRAIN_PATH//$'\r'/}"
VAL_PATH="${VAL_PATH//$'\r'/}"
TEST_PATH="${TEST_PATH//$'\r'/}"

# Define model parameters
SDE="subvp"
N_ITERS="380000"
CKP_PATH=''

# Define training parameters
episodes=10
epochs=10
modality="generation"
weight_method="v2"
lr=0.0001
wd=0.01

# Loop through methods, shots, and additional parameters
for method in {'protonet','covnet','meta_deepbdc'}; do 
  for shot in {1,2,3,4,5}; do
    for n_add in {1,2,3}; do
      if [[ $modality == "generation" ]]; then
        echo "Generation modality"
        n_add=1
        OUTPUT_PATH=''
      else
        echo "Baseline modality"
        n_add=0
        OUTPUT_PATH=''
      fi

      # Remove carriage return characters from output path
      OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
      mkdir -p $OUTPUT_PATH

      # Run the training script
      /home/eva.pachetti/miniconda3/envs/evaenv/bin/python meta_train.py --method $method --output_path "$OUTPUT_PATH" --image_size 128 \
          --learning_rate $lr --weight_decay $wd --epoch $epochs --margin 0.5 --milestones 40 --n_shot $shot --n_query 5 --train_n_way 4 \
          --val_n_way 4 --num_classes 7 --train_n_episode $episodes --val_n_episode $episodes --reduce_dim 256 --csv_path_train "$TRAIN_PATH" \
          --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --num_scales 1000 --sde "$SDE" \
          --beta_min 0.1 --beta_max 20 --pretrained_score_model --ckp_path "$CKP_PATH" --weight_method $weight_method --n_add $n_add --generation

      # Define checkpoint paths
      SCORE_CKP_PATH=$OUTPUT_PATH/score_checkpoint.pth
      CLS_CKP_PATH=$OUTPUT_PATH/cls_checkpoint.pth

      # Run the testing script
      /home/eva.pachetti/miniconda3/envs/evaenv/bin/python test.py --method $method --output_path "$OUTPUT_PATH" --image_size 128 \
          --n_shot $shot --n_query 0 --test_n_way 4 --test_n_episode $episodes --csv_path_train "$TRAIN_PATH" \
          --csv_path_val "$VAL_PATH" --csv_path_test "$TEST_PATH" --num_scales 1000 --sde "$SDE" \
          --beta_min 0.1 --beta_max 20 --score_ckp_path "$SCORE_CKP_PATH" --cls_ckp_path "$CLS_CKP_PATH" \
          --weight_method $weight_method --n_add $n_add --generation 
    done
  done
done
