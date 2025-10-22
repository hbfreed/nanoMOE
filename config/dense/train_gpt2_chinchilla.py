# GPT-2 dense model configuration for OpenWebText training
# Chinchilla-optimal scaling

dataset = "openwebtext"

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# MoE configuration (disabled for dense)
use_moe = False

out_dir = "out-openwebtext/dense"

wandb_log = True
wandb_project = "gpt2-chinchilla"
wandb_run_name = "dense"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 39 gradaccum * 3 GPUs = 1,445,888
batch_size = 12
n_ctx = 1024
gradient_accumulation_steps = 13 * 3

# 5217 gets us to roughly 2.5 billion tokens, which is chinchilla optimal for a model of this size (125m * 20 = 2.5 billion)
max_iters = 5217
lr_decay_iters = 5217

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate
learning_rate = 6e-4
min_lr = 6e-5
