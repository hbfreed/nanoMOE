#!/bin/bash

# Extreme low sweep: 1e-9 to 1e-6
# Cover all orders of magnitude below what we've tested
# Find the true natural equilibrium

PROJECT="wikitext-compute-sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================="
echo "EXTREME LOW SWEEP: 1e-9 to 1e-6"
echo "Finding the natural floor"
echo "========================================="

# Just hit every order of magnitude
LBL_WEIGHTS=(0)
FLOPS_WEIGHTS=(0.000000001 0.00000001 0.0000001 0.000001)
# That's:      1e-9         1e-8        1e-7      1e-6

for lbl in "${LBL_WEIGHTS[@]}"; do
    for flops in "${FLOPS_WEIGHTS[@]}"; do
        echo "========================================="
        echo "Training with LBL=$lbl, FLOPs=$flops"
        echo "========================================="

        torchrun --standalone --nproc_per_node=3 train.py \
            config/moe_variable/train_moe_wikitext_variable.py \
            --wandb_project="$PROJECT" \
            --wandb_run_name="lbl_w${lbl}_flops_w${flops}_${TIMESTAMP}" \
            --load_balance_loss_weight=$lbl \
            --compute_loss_weight=$flops \
            --out_dir="out-lbl-w${lbl}-flops-w${flops}-${TIMESTAMP}"

        echo ""
        echo "Completed training with LBL=$lbl, FLOPs=$flops"
        echo ""
    done
done

echo "All training runs complete!"
echo "Total: 4 runs"
echo ""
echo "Expected: As FLOPs weight decreases, big expert usage should INCREASE"
echo "Looking for the true 'no pressure' equilibrium"
echo "If we find a sweet spot, we can do fine-grained sweep around it"
