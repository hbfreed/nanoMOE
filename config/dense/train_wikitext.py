# Dense model configuration for WikiText-103 training

dataset = "wikitext"

# Model configuration
n_layer = 8
n_head = 8
n_embd = 640
dropout = 0.2
bias = False

# MoE configuration (disabled for dense)
use_moe = False

out_dir = "out-wikitext/dense"

wandb_log = True
wandb_project = "moe-wikitext"
wandb_run_name = "dense"

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
