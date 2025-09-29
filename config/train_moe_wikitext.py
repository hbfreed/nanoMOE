# MoE model configuration for WikiText-103 training
dataset = "wikitext"

n_layer = 8
n_head = 8
n_embd = 640
dropout = 0.2
bias = False

# MoE configuration
use_moe = True 
num_experts = 32 
num_experts_per_tok = 4
norm_topk_prob = True
block_size = 128
block_k = 32

wandb_log = True
wandb_project = 'moe-wikitext'
wandb_run_name = f'moe-{num_experts}-{num_experts_per_tok}-{n_head}h-{n_layer}l'

# Set output directory based on MoE configuration
if use_moe:
    out_dir = f'out-wikitext/{num_experts}_{num_experts_per_tok}_{n_head}h_{n_layer}l'
else:
    out_dir = 'out-wikitext/dense'

batch_size = 32 # need to do bs 32 with 6 grad accum to fit on the gpus 
n_ctx = 1024
gradient_accumulation_steps = 6 

max_iters = 5000
lr_decay_iters = 5000

# Eval
eval_interval = 250
eval_iters = 200
log_interval = 10

# Optimization
weight_decay = 1e-1
learning_rate = 3e-4
min_lr = 3e-5
