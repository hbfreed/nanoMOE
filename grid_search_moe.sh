#!/bin/bash

# Grid search over MoE hyperparameters on WikiText
# Total experts: 8, 16, 32, 64
# Active experts: 2, 4, 8
# Plus a dense baseline (no MoE)

# Create arrays for the grid search
total_experts=(8 16 32 64)
active_experts=(2 4 8)

# Keep track of which GPU to use
gpu_idx=0

# Create a temporary file to track running processes
pidfile=$(mktemp)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting MoE grid search on WikiText..."
echo "Using GPUs 0, 1, 2"
echo ""

# First, launch the dense baseline
wandb_name="dense-wikitext-baseline"
echo "Launching dense baseline on GPU $gpu_idx (wandb: $wandb_name)"

CUDA_VISIBLE_DEVICES=$gpu_idx python3 train.py \
    --dataset=wikitext \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --dropout=0.2 \
    --bias=False \
    --batch_size=64 \
    --n_ctx=256 \
    --gradient_accumulation_steps=1 \
    --max_iters=5000 \
    --lr_decay_iters=5000 \
    --learning_rate=6e-4 \
    --min_lr=6e-5 \
    --eval_interval=250 \
    --eval_iters=200 \
    --wandb_log=True \
    --wandb_project="moe-wikitext-gridsearch" \
    --wandb_run_name="$wandb_name" \
    > logs/${wandb_name}.log 2>&1 &

# Save the PID
echo $! >> $pidfile
gpu_idx=$(( (gpu_idx + 1) % 3 ))

# Sleep for 10 seconds to allow GPU memory to be properly allocated
sleep 10

# Now run the MoE configurations
for total in "${total_experts[@]}"; do
    for active in "${active_experts[@]}"; do
        # Skip invalid combinations (active > total)
        if [ $active -gt $total ]; then
            echo "Skipping invalid combo: total=$total, active=$active (active > total)"
            continue
        fi

        # Skip the 8/8 combination as requested
        if [ $total -eq 8 ] && [ $active -eq 8 ]; then
            echo "Skipping combo: total=8, active=8 (as requested)"
            continue
        fi

        # Create wandb run name
        wandb_name="moe-wikitext-experts${total}-active${active}"

        echo "Launching: total=$total, active=$active on GPU $gpu_idx (wandb: $wandb_name)"

        # Launch the training run in the background
        CUDA_VISIBLE_DEVICES=$gpu_idx python3 train.py \
            --dataset=wikitext \
            --n_layer=6 \
            --n_head=6 \
            --n_embd=384 \
            --dropout=0.2 \
            --bias=False \
            --batch_size=64 \
            --n_ctx=256\
            --gradient_accumulation_steps=1 \
            --max_iters=5000 \
            --lr_decay_iters=5000 \
            --learning_rate=6e-4 \
            --min_lr=6e-5 \
            --eval_interval=250 \
            --eval_iters=200 \
            --use_moe=True \
            --num_experts=$total \
            --num_experts_per_tok=$active \
            --norm_topk_prob=True \
            --block_size=64 \
            --block_k=64 \
            --wandb_log=True \
            --wandb_project="moe-wikitext-gridsearch" \
            --wandb_run_name="$wandb_name" \
            > logs/${wandb_name}.log 2>&1 &

        # Save the PID
        echo $! >> $pidfile

        # Cycle through GPUs (0, 1, 2)
        gpu_idx=$(( (gpu_idx + 1) % 3 ))

        # Sleep for 10 seconds to allow GPU memory to be properly allocated
        sleep 10

        # If we've launched 3 jobs, wait for one to finish before continuing
        if [ $(wc -l < $pidfile) -eq 3 ]; then
            echo "All 3 GPUs in use, waiting for a job to complete..."

            # Wait for any of the background jobs to finish
            wait -n $(cat $pidfile)

            # Clean up finished PIDs from the file
            > $pidfile.tmp
            while IFS= read -r pid; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo "$pid" >> $pidfile.tmp
                fi
            done < $pidfile
            mv $pidfile.tmp $pidfile
        fi
    done
done

# Wait for all remaining jobs to complete
echo ""
echo "All jobs launched. Waiting for remaining jobs to complete..."
wait $(cat $pidfile)

# Clean up
rm -f $pidfile

echo ""
echo "Grid search complete!"
echo "Total runs: 1 dense + 9 MoE configurations = 10 runs"