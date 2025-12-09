# MoE model configuration for WikiText-103 training

dataset = "wikitext"

# Model configuration
n_layer = 8
n_head = 8
n_embd = 640
dropout = 0.2
bias = False

# MoE configuration
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128
expert_sizes = [(num_experts, n_embd * 4 // num_experts_per_tok)]
load_balance_loss_weight = 0.01
compute_loss_weight = 0.0

# Create string representation of expert sizes for naming
expert_sizes_str = "-".join([f"{h}x{d}" for h, d in expert_sizes])

out_dir = f"out-wikitext/moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}"

wandb_log = True
wandb_project = "wikitext"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}"

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
