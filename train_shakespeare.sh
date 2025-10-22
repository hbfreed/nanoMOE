#!/bin/bash

# Train Shakespeare models - all 3 configs in parallel on separate GPUs
# Dense on GPU 0, MoE uniform on GPU 1, MoE variable on GPU 2

echo "Starting Shakespeare training on 3 GPUs in parallel..."

# Dense model on GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py config/dense/train_shakespeare_char.py &
PID1=$!
echo "Started dense training on GPU 0 (PID: $PID1)"

# MoE uniform on GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py config/moe/train_moe_shakespeare_char.py &
PID2=$!
echo "Started MoE uniform training on GPU 1 (PID: $PID2)"

# MoE variable on GPU 2
CUDA_VISIBLE_DEVICES=2 python train.py config/moe_variable/train_moe_shakespeare_char_variable.py &
PID3=$!
echo "Started MoE variable training on GPU 2 (PID: $PID3)"

# Wait for all processes to complete
echo "Waiting for all training jobs to complete..."
wait $PID1
echo "Dense training completed"
wait $PID2
echo "MoE uniform training completed"
wait $PID3
echo "MoE variable training completed"

echo "All Shakespeare training jobs completed!"
