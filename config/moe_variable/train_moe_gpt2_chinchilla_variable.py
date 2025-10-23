# GPT-2 MoE model configuration for OpenWebText training
# with VARIABLE-SIZE EXPERTS - Chinchilla-optimal scaling
# 4 large experts (2944) and 4 small experts (128)

dataset = "openwebtext"

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# MoE configuration with VARIABLE-SIZE EXPERTS
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128
block_k = 64
expert_sizes = [(4, 2944), (4, 128)]  # 4 large (2944) + 4 small (128)
load_balance_loss_weight = 0.5

# Create string representation of expert sizes for naming
expert_sizes_str = "-".join([f"{h}x{d}" for h, d in expert_sizes])

out_dir = f"out-openwebtext/moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"

wandb_log = True
wandb_project = "gpt2-chinchilla"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"

# these make the total batch size be ~0.5M
batch_size = 12
n_ctx = 1024
gradient_accumulation_steps = 13 * 3

# gets us to roughly 2.5 billion tokens, which is chinchilla optimal for a model of this size (125m * 20 = 2.5 billion)
max_iters = 5216
lr_decay_iters = 5216

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate - slightly higher for MoE
learning_rate = 6e-4
min_lr = 6e-5
