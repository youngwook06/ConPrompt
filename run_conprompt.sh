#!/bin/bash

##############credits###############
# https://github.com/princeton-nlp/SimCSE
####################################


# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

TRAIN_FILE_NAME=conprompt_pre-train_dataset
NUM_GPU=4

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID conprompt.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/$TRAIN_FILE_NAME.csv \
    --output_dir result/ToxiGen-ConPrompt \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --evaluation_strategy no \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.03 \
    --do_train \
    --fp16 \
    --do_mlm \
    "$@"
