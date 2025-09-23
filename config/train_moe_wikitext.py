# MoE model configuration for WikiText-103 training
wandb_log = True
wandb_project = 'moe-wikitext'
wandb_run_name = 'moe-64-8a'
dataset = "wikitext"

n_layer = 6 
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# MoE configuration
use_moe = True
num_experts = 64
num_experts_per_tok = 8
norm_topk_prob = True
block_size = 128
block_k = 32

batch_size = 64
n_ctx = 256
gradient_accumulation_steps = 3 

max_iters = 5000
lr_decay_iters = 5000

# Eval
eval_interval = 250
eval_iters = 200
log_interval = 10

# Optimization
weight_decay = 1e-1
learning_rate = 1e-3
min_lr = 1e-4
