#!/bin/bash

export DATA_DIR=directory_name
export TRAIN_FILE_NAME=train.csv
export TEST_FILE_NAME=test.csv
export OUTPUT_DIR=${DATA_DIR}/output

export MAX_LENGTH=64
export BATCH_SIZE=32
export NUM_EPOCHS=3
export LEARING_RATE=2e-5
export SAVE_STEPS=50000
export EVAL_STEPS=5000
export SEED=1

python src\run_glue.py \
    --train_file ${DATA_DIR}/${TRAIN_FILE_NAME} \
    --validation_file ${DATA_DIR}/${TEST_FILE_NAME} \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --max_seq_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --save_steps ${SAVE_STEPS} \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --fp16
