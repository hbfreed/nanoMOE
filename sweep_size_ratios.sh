#!/bin/bash

# Capacity-matched size ratio sweep with FLOPs loss
# Total capacity: 8 × 1280 = 10,240 parameters per layer
# Testing which size ratio gives best efficiency/quality tradeoff
# Using FLOPs=0.004 (previous best) and LBL=0.0 (no constraint, pure efficiency)

PROJECT="wikitext-size-ratio-sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="wikitext_experiments/size_ratio_sweep_${TIMESTAMP}"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR/logs"

# Format: "ratio:large_size:small_size"
# Note: Sizes must be multiples of 128
declare -a SIZE_CONFIGS=(
    "1:1280:1280"
    "2:1664:896"      # 2:1 ratio (~1.86:1 actual) - total: 10,240
    "3:1920:640"      # 3:1 ratio (exact 3:1) - total: 10,240
    "4:2048:512"      # 4:1 ratio (exact 4:1) - total: 10,240
    "6:2176:384"      # 6:1 ratio (~5.67:1 actual) - total: 10,240
    "9:2304:256"     # 9:1 ratio (exact 9:1) - total: 10,240
    "19:2432:128"     # 19:1 ratio (exact 19:1) - total: 10,240
)

# Fixed hyperparameters - using previous best FLOPs weight
LBL_WEIGHT=0.01    # LBL balances *within* expert groups
FLOPS_WEIGHT=0.004  # Previous best FLOPs weight from 19:1 experiments

echo "Starting capacity-matched size ratio sweep at $TIMESTAMP"
echo "Project: $PROJECT"
echo "LBL weight: $LBL_WEIGHT (fixed)"
echo "FLOPs weight: $FLOPS_WEIGHT (fixed)"
echo ""

for config in "${SIZE_CONFIGS[@]}"; do
    IFS=':' read -r ratio large_size small_size <<< "$config"

    RUN_NAME="ratio${ratio}_lbl${LBL_WEIGHT}_flops${FLOPS_WEIGHT}_${TIMESTAMP}"

    echo "========================================"
    echo "Running: ${ratio}:1 ratio"
    echo "Large experts (4x): $large_size"
    echo "Small experts (4x): $small_size"
    echo "Total capacity: $((4 * large_size + 4 * small_size))"
    echo "Run name: $RUN_NAME"
    echo "========================================"

    torchrun --standalone --nproc_per_node=3 train.py config/moe_variable/train_moe_wikitext_variable.py \
        --wandb_project="$PROJECT" \
        --wandb_run_name="$RUN_NAME" \
        --load_balance_loss_weight=$LBL_WEIGHT \
        --compute_loss_weight=$FLOPS_WEIGHT \
        --expert_sizes="[(4, $large_size), (4, $small_size)]" \
        --out_dir="${EXPERIMENT_DIR}/${RUN_NAME}" \
        2>&1 | tee "${EXPERIMENT_DIR}/logs/${RUN_NAME}.log"

    echo ""
    echo "Completed: ${ratio}:1 ratio"
    echo ""

    # Small delay between runs
    sleep 5
done

echo "All size ratio sweep runs completed!"
echo "Check wandb project: $PROJECT"
