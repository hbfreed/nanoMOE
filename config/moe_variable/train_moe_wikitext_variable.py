# MoE model configuration for WikiText-103 training
# with VARIABLE-SIZE EXPERTS
# 4 large experts (2432) and 4 small experts (128)

dataset = "wikitext"

# Model configuration
n_layer = 8
n_head = 8
n_embd = 640
dropout = 0.2
bias = False

# MoE configuration with VARIABLE-SIZE EXPERTS
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128
block_k = 32
expert_sizes = [(4, 2432), (4, 128)]  # 4 large (2432) + 4 small (128)
load_balance_loss_weight = 0.5

# Create string representation of expert sizes for naming
expert_sizes_str = "-".join([f"{h}x{d}" for h, d in expert_sizes])

out_dir = (
    f"out-wikitext/moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"
)

wandb_log = True
wandb_project = "wikitext"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"

# these make the total batch size be 32 * 1024 * 6 = 196,608
batch_size = 32
n_ctx = 1024
gradient_accumulation_steps = 6

max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate
learning_rate = 3e-4
min_lr = 3e-5
