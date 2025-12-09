# Train a miniature character-level shakespeare model with MoE

dataset = "shakespeare_char"

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

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

out_dir = (
    f"out-shakespeare-char/moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}"
)

wandb_log = True
wandb_project = "shakespeare-char"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}"

# these make the total batch size
batch_size = 64
n_ctx = 256
gradient_accumulation_steps = 1

max_iters = 2500
lr_decay_iters = 2500

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# learning rate - with baby networks can afford to go a bit higher
learning_rate = 1e-3
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
