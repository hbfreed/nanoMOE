# GPT-2 MoE model configuration for OpenWebText training
# with VARIABLE-SIZE EXPERTS - Chinchilla-optimal scaling

dataset = "openwebtext"

# Model configuration - ~250M active 
n_layer = 16 
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

# MoE configuration with VARIABLE-SIZE EXPERTS
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128
block_k = 64
expert_sizes = [(4, 3456), (4, 640)]
load_balance_loss_weight = 0.01
router_z_loss_weight = 0.001
compute_loss_weight = 0.004

# Create string representation of expert sizes for naming
expert_sizes_str = "-".join([f"{h}x{d}" for h, d in expert_sizes])

out_dir = f"out-openwebtext/moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"

wandb_log = True
wandb_project = "gpt2-250-chinchilla"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-variable-{expert_sizes_str}"

# these make the total batch size be ~0.5M
batch_size = 8
n_ctx = 1024
gradient_accumulation_steps = 60

# gets us to roughly 5 billion tokens, which is chinchilla optimal for a model of this size (250m * 20 = 5 billion)
max_iters = 10500 
lr_decay_iters = 10500

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate - slightly higher for MoE
learning_rate = 5e-4
min_lr = 5e-5
