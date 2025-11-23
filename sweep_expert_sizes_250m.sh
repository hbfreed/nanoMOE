#!/bin/bash

# Expert size configuration sweep for GPT-2 250M Chinchilla
# Capacity-matched: 16,384 parameters per MoE layer
# Testing: 5:1 ratio, 31:1 ratio, and 1:1 uniform baseline

PROJECT="gpt2-250m-expert-sizes"
# Resume from existing experiment directory (comment out to start fresh)
EXPERIMENT_DIR="gpt2_250m_experiments/expert_sizes_20251106_090944"
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# EXPERIMENT_DIR="gpt2_250m_experiments/expert_sizes_${TIMESTAMP}"

# Create experiment directory if needed
mkdir -p "$EXPERIMENT_DIR/logs"

# Expert size configurations
# Format: "name:expert_sizes_arg"
declare -a SIZE_CONFIGS=(
    "5to1:[(4, 3456), (4, 640)]"      # 5:1 ratio - 4 large + 4 small
    "31to1:[(4, 3968), (4, 128)]"     # 31:1 ratio - 4 large + 4 small
    "uniform:[(8, 2048)]"              # 1:1 baseline - 8 uniform experts
)

# Fixed hyperparameters from config
LBL_WEIGHT=0.01
COMPUTE_WEIGHT=0.004
SEED=1337

echo "========================================"
echo "Starting expert size sweep (RESUMING from 20251106_090944)"
echo "Project: $PROJECT"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "LBL weight: $LBL_WEIGHT"
echo "Compute weight: $COMPUTE_WEIGHT"
echo "Seed: $SEED"
echo "Running 3 configs in parallel (1 per GPU)"
echo "========================================"
echo ""

# Launch all three configs in parallel, one per GPU
GPU_ID=0
RESUME_TIMESTAMP="20251106_090944"  # Timestamp from existing checkpoints
for config in "${SIZE_CONFIGS[@]}"; do
    IFS=':' read -r name expert_sizes <<< "$config"

    RUN_NAME="sizes_${name}_lbl${LBL_WEIGHT}_compute${COMPUTE_WEIGHT}_seed${SEED}_${RESUME_TIMESTAMP}"

    echo "========================================"
    echo "Launching: $name configuration on GPU $GPU_ID"
    echo "Expert sizes: $expert_sizes"
    echo "Run name: $RUN_NAME"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py config/moe_variable/train_moe_gpt2_250m_chinchilla_variable.py \
        --wandb_project="$PROJECT" \
        --wandb_run_name="$RUN_NAME" \
        --load_balance_loss_weight=$LBL_WEIGHT \
        --compute_loss_weight=$COMPUTE_WEIGHT \
        --seed=$SEED \
        --expert_sizes="$expert_sizes" \
        --out_dir="${EXPERIMENT_DIR}/${RUN_NAME}" \
        --init_from=resume \
        2>&1 | tee "${EXPERIMENT_DIR}/logs/${RUN_NAME}.log" &

    echo "Launched on GPU $GPU_ID (PID: $!)"
    echo ""

    GPU_ID=$((GPU_ID + 1))
done

echo "All jobs launched in parallel. Waiting for completion..."
wait

echo ""
echo "All jobs completed!"

echo "========================================"
echo "All expert size sweep runs completed!"
echo "Total configurations tested: ${#SIZE_CONFIGS[@]}"
echo "Check wandb project: $PROJECT"
echo "========================================"
