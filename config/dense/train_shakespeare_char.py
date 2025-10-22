# Train a miniature character-level shakespeare model (dense)

dataset = "shakespeare_char"

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# MoE configuration (disabled for dense)
use_moe = False

out_dir = "out-shakespeare-char/dense"

wandb_log = True
wandb_project = "shakespeare-char"
wandb_run_name = "dense"

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
