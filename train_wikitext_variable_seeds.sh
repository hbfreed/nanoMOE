#!/bin/bash

# Train GPT-2 WikiText Variable MoE with 3 different random seeds
# Each training uses all 3 GPUs via DDP

SEEDS=(42 1337 1223)
MODEL_NAME="moe-8x2-variable-4x2432-4x128"

echo "Starting Wikitext Variable MoE training with 3 different seeds..."

for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Training with seed: $SEED"
    echo "========================================"
    torchrun --standalone --nproc_per_node=3 train.py \
        config/moe_variable/train_moe_wikitext_variable.py \
        --seed=$SEED \
        --out_dir="out-wikitext/${MODEL_NAME}-seed${SEED}" \
        --wandb_run_name="${MODEL_NAME}-seed${SEED}"

    echo "Training with seed $SEED completed"
    echo ""
done

echo "========================================"
echo "All seed runs completed!"
echo "========================================"
