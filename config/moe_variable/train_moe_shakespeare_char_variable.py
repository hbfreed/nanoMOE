# train a miniature character-level shakespeare model
# with VARIABLE-SIZE EXPERTS!
# 4 large experts (768) and 4 small experts (384)

out_dir = 'out-moe-shakespeare-char-variable'
eval_interval = 250 # keep frequent because we'll overfit, especially moes
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-eval'
wandb_run_name = 'mini-gpt-moe-variable-8x2-4large4small'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
n_ctx = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2500
lr_decay_iters = 2500 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# MoE configuration with VARIABLE-SIZE EXPERTS
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128  # Triton kernel block size
block_k = 128     # Triton kernel K dimension

expert_sizes = [(4, 1408), (4, 128)]
