#!/bin/bash

# Train WikiText models - all 3 configs sequentially with torchrun on 3 GPUs
# Each training uses all 3 GPUs via DDP

echo "Starting WikiText training with torchrun (3 GPUs per run)..."

# Dense model
echo "========================================"
echo "Training dense model..."
echo "========================================"
torchrun --standalone --nproc_per_node=3 train.py config/dense/train_wikitext.py
echo "Dense training completed"

# MoE uniform
echo "========================================"
echo "Training MoE uniform model..."
echo "========================================"
torchrun --standalone --nproc_per_node=3 train.py config/moe/train_moe_wikitext.py
echo "MoE uniform training completed"

# MoE variable
echo "========================================"
echo "Training MoE variable model..."
echo "========================================"
torchrun --standalone --nproc_per_node=3 train.py config/moe_variable/train_moe_wikitext_variable.py
echo "MoE variable training completed"

echo "========================================"
echo "All WikiText training jobs completed!"
echo "========================================"
