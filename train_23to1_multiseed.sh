#!/bin/bash

# Train 23:1 ratio (2944:128) with multiple seeds
# Testing consistency of routing patterns across different initializations
# Using LBL=0.01 and compute=0.004

PROJECT="gpt2-multiseed-23to1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="gpt2_experiments/multiseed_23to1_${TIMESTAMP}"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR/logs"

# Fixed configuration
LARGE_SIZE=2944
SMALL_SIZE=128
LBL_WEIGHT=0.01
COMPUTE_WEIGHT=0.004

# Seeds to test
SEEDS=(42 1223)

echo "========================================="
echo "Multi-seed training for 23:1 ratio"
echo "Project: $PROJECT"
echo "Timestamp: $TIMESTAMP"
echo "Large experts: 4 × $LARGE_SIZE"
echo "Small experts: 4 × $SMALL_SIZE"
echo "LBL weight: $LBL_WEIGHT"
echo "Compute weight: $COMPUTE_WEIGHT"
echo "Seeds: ${SEEDS[@]}"
echo "========================================="
echo ""

for seed in "${SEEDS[@]}"; do
    RUN_NAME="ratio23_lbl${LBL_WEIGHT}_compute${COMPUTE_WEIGHT}_seed${seed}_${TIMESTAMP}"

    echo "========================================="
    echo "Training with seed=$seed"
    echo "Run name: $RUN_NAME"
    echo "========================================="

    torchrun --standalone --nproc_per_node=3 train.py config/moe_variable/train_moe_gpt2_chinchilla_variable.py \
        --wandb_project="$PROJECT" \
        --wandb_run_name="$RUN_NAME" \
        --load_balance_loss_weight=$LBL_WEIGHT \
        --compute_loss_weight=$COMPUTE_WEIGHT \
        --seed=$seed \
        --expert_sizes="[(4, $LARGE_SIZE), (4, $SMALL_SIZE)]" \
        --out_dir="${EXPERIMENT_DIR}/${RUN_NAME}" \
        2>&1 | tee "${EXPERIMENT_DIR}/logs/${RUN_NAME}.log"

    echo ""
    echo "Completed training with seed=$seed"
    echo ""

    # Small delay between runs
    sleep 5
done

echo "All multi-seed training runs completed!"
echo "Total runs: ${#SEEDS[@]}"
echo "Check wandb project: $PROJECT"
