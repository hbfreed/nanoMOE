#!/bin/bash

# Capacity-matched size ratio sweep with compute loss for GPT-2
# Total capacity: 8 × 1536 = 12,288 parameters per layer
# Testing which size ratio gives best efficiency/quality tradeoff
# Using compute=0.004 (from wikitext best) and LBL=0.0 (no constraint, pure efficiency)

PROJECT="gpt2-size-ratio-sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="gpt2_experiments/size_ratio_sweep_${TIMESTAMP}"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR/logs"

# Expert size configurations (4 large + 4 small = 12,288 total)
# Format: "ratio:large_size:small_size"
# Note: Sizes must be multiples of 128
declare -a SIZE_CONFIGS=(
    "1:1536:1536"     # 1:1 ratio (uniform) - total: 12,288
    # "2:2048:1024"     # 2:1 ratio (exact 2:1) - total: 12,288
    # "3:2304:768"      # 3:1 ratio (exact 3:1) - total: 12,288
    "5:2560:512"      # 5:1 ratio (exact 5:1) - total: 12,288
    "7:2688:384"      # 7:1 ratio (exact 7:1) - total: 12,288
    "11:2816:256"     # 11:1 ratio (exact 11:1) - total: 12,288
    "23:2944:128"     # 23:1 ratio (exact 23:1) - total: 12,288
)

# Fixed hyperparameters - using previous best compute weight from wikitext
LBL_WEIGHT=0.01
COMPUTE_WEIGHT=0.004  # Previous best compute weight from wikitext experiments
SEED=1337           # Fixed seed for reproducibility

echo "Starting capacity-matched size ratio sweep at $TIMESTAMP"
echo "Project: $PROJECT"
echo "LBL weight: $LBL_WEIGHT (fixed)"
echo "Compute weight: $COMPUTE_WEIGHT (fixed)"
echo ""

for config in "${SIZE_CONFIGS[@]}"; do
    IFS=':' read -r ratio large_size small_size <<< "$config"

    RUN_NAME="ratio${ratio}_lbl${LBL_WEIGHT}_compute${COMPUTE_WEIGHT}_seed${SEED}_${TIMESTAMP}"

    echo "========================================"
    echo "Running: ${ratio}:1 ratio"
    echo "Large experts (4x): $large_size"
    echo "Small experts (4x): $small_size"
    echo "Total capacity: $((4 * large_size + 4 * small_size))"
    echo "Run name: $RUN_NAME"
    echo "========================================"

    torchrun --standalone --nproc_per_node=3 train.py config/moe_variable/train_moe_gpt2_chinchilla_variable.py \
        --wandb_project="$PROJECT" \
        --wandb_run_name="$RUN_NAME" \
        --load_balance_loss_weight=$LBL_WEIGHT \
        --compute_loss_weight=$COMPUTE_WEIGHT \
        --seed=$SEED \
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
