"""This really only works with one batch of texts rn"""

import json
from collections import Counter, defaultdict

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPTConfig, GPTWithTracking, MoeMLPWithTracking

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # Or 'medium' for even more speed

config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    bias=False,
    vocab_size=50304,
    use_moe=True,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    block_size=128,
    block_k=64,
    # expert_sizes=[(4, 2560), (4, 512)],  # 5:1
    expert_sizes=[(4, 2944), (4, 128)],  # 23:1
)

checkpoint_path = (
    "gpt2_experiments/multiseed_23to1/ratio23_lbl0.01_compute0.004_seed1223/ckpt.pt"
    # "gpt2_experiments/multiseed_5to1/ratio5_lbl0.01_compute0.004_seed1223/ckpt.pt"
)

model = GPTWithTracking(config).to(torch.bfloat16)

for block in model.transformer.h:
    if hasattr(block.mlp, "expert_sizes"):
        old_mlp = block.mlp
        block.mlp = MoeMLPWithTracking(config).to(torch.bfloat16)
        block.mlp.load_state_dict(old_mlp.state_dict())

checkpoint = torch.load(checkpoint_path, map_location="cpu")

state_dict = checkpoint["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

tokenizer = tiktoken.get_encoding("gpt2")

device = "cuda"
model = model.to("cuda")
model.eval()

num_layers = config.n_layer
expert_sizes = model.transformer.h[0].mlp.expert_sizes

# initalize our tracking structures
token_stats_per_layer = {}
token_combinations_per_layer = {}
for layer_idx in range(num_layers):
    layer_name = f"layer_{layer_idx}"
    token_stats_per_layer[layer_name] = defaultdict(
        lambda: {
            "expert_counts": np.zeros(config.num_experts, dtype=np.int64),
            "total_occurrences": 0,
            "total_entropy": 0.0,
            "total_layer_entropy": 0.0,
            "expert_size_sum": 0.0,
        }
    )
    token_combinations_per_layer[layer_name] = defaultdict(Counter)

batch_size = 1
seq_len = 1024

eval_texts = """**Burgundy**

    Every time I drink Burgundy wine, I feel like I’m part of a secret club of people who know things. The best things. Like Star Trek things. Did you know that Burgundy was drunk in space? Yep, it was. Nuits-Saint-Georges to be exact.

    If I scrape my knee, I immediately consider pouring Burgundy on it. I’m sold on its healing qualities. If I’m at home sick with a cold, I think to drink it. If I see someone with acne, I want to rub it on his or her face. I want to bathe my daughter in it. Red Burgundy—Pinot Noir in its purest form—goes with absolutely everything: breakfast, lunch, and dinner. It goes with fast food, birthday food, food that other chefs cook for you, and fries. At Joe Beef, we’ve never met a bottle of Burgundy that we didn’t like; even the ones that apparently suck. My judgment is so flawed that I can even say positive things about the worst Burgundian battery acid.

    The Burgundy drinker is a great lover, is well-read, and has impeccable style. He or she loves woodland animals and is the kind of person who might have a miniature bear as a pet (like Charles McKiernan). He likely owns several tweed coats and partitions his garden with stone walls. Her prized possession may be a mechanic’s nightmare of an old Peugeot. When drinking any other wine, the Burgundy drinker typically says insane things like, “this Sancerre is quite Burgundian,” or “this Bandol reminds me of why I love Burgundy!” Drinkers of Burgundy scold themselves for overlooking (abandoning!) certain villages: “We really do not drink enough Pernand-Vergelesses,” or “I simply must rediscover Saint-Aubin!” Of course, this is all fiction, but I always seem to find something in a bottle that expresses a rural sense of pensiveness. For me, it’s like drinking archeology.

    It is said that Celts grew vines in Burgundy as early as 51 B.C. Later, Benedictine monks and Cisterians set about the task of separating vineyard plots because they believed certain areas provided consistently different wines. They set up the system of _crus_ and the notion of terroir. I can feel the mystical presence of these snail eaters when I walk through Burgundian vineyards that are just off the beaten path. Sometimes when wine reps want us to try a $58 Oregon Pinot Noir, the answer is an understandable, “No thanks.”

    New World Pinot, out of respect, shouldn’t be more expensive than true noble Burgundy. Hundreds of years ago everyone agreed that this bacon strip–shaped piece of land that runs just south of Dijon is the best place on earth for these grapes to exist. It’s a World Heritage site, meaning the laws are so strict that you can’t irrigate—in other words, you can’t just run the hose into a vineyard.

    In my dream-world restaurant, I would serve only red and white Burgundies. From Dijon all the way to the Beaujolais, each village in Burgundy is a treasure. Each day in Burgundy, people ask, “What did you eat for breakfast?” Then, “What did you eat for lunch?” Like Montrealers, these people are obsessed with food and wine. (Wouldn’t you love a Romanée Conti or Meursault Perrières sweatshirt? Or, a “My parents came back from Burgundy and all I got was this shitty Clos Vougeot” T-shirt?)

    A plethora of French winemakers visit Montreal on a regular basis—Étienne de Montille, Pierre-Yves Colin-Morey, the Muzard brothers—and most seem to have the impression that Quebecers are a bunch of cowboys riding around on skidoos, fishing giant salmon, and shooting grizzly bears. This isn’t true, of course (though I have to say, I didn’t see one bird for two years while living in Burgundy; they’ve eaten every fucking thing that moves there). Always seen as the exotic cousins to the French, Quebec may very well be the last frontier of moose hunting, angling, foraging, and sugar shacks.
"""


tokenized_texts = tokenizer.encode(eval_texts)
total_tokens = len(tokenized_texts)
num_batches = total_tokens // seq_len + 1

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * seq_len
    end_idx = start_idx + seq_len
    batch_tokens = torch.as_tensor(
        tokenized_texts[start_idx:end_idx], dtype=torch.long, device=device
    )
    if batch_tokens.size(0) < seq_len:
        batch_tokens = F.pad(
            batch_tokens, (0, seq_len - batch_tokens.size(0)), value=tokenizer.eot_token
        )
    batch_tokens = batch_tokens.unsqueeze(0)

    with torch.inference_mode():
        logits, loss, aux_loss = model(batch_tokens, targets=batch_tokens)
    output_probs = F.softmax(logits[0], dim=-1)
    epsilon = 1e-10  # epsilon so we don't log(0) for entropy
    output_entropy = (
        -(output_probs * torch.log(output_probs + epsilon))
        .sum(dim=-1)
        .float()
        .cpu()
        .numpy()
    )

# sizing_map = {
#     0: 2560,
#     1: 2560,
#     2: 2560,
#     3: 2560,
#     4: 512,
#     5: 512,
#     6: 512,
#     7: 512,
# }  # for 5:1
sizing_map = {
    0: 2944,
    1: 2944,
    2: 2944,
    3: 2944,
    4: 128,
    5: 128,
    6: 128,
    7: 128,
}  # for 23:1
#
token_data = {"text": eval_texts, "tokens": []}
for token_idx, token in enumerate(tokenized_texts):
    layers_list = []
    total_expert_size = 0

    for layer_name, assignments in aux_loss["expert_assignments"].items():
        layer_idx = int(layer_name.split("_")[1])
        expert_ids = assignments[0, token_idx].tolist()
        expert_sizes = [sizing_map[expert_ids[0]], sizing_map[expert_ids[1]]]
        total_expert_size += sum(expert_sizes)
        layers_list.append(
            {
                "layer": layer_idx,
                "expert_ids": expert_ids,
                "expert_sizes": expert_sizes,
            }
        )

    intermediate_dict = {
        "token": tokenizer.decode([token]),
        "position": token_idx,
        "mean_expert_size": round(total_expert_size / config.n_layer, 2),
        "layers": layers_list,
    }
    token_data["tokens"].append(intermediate_dict)

with open("routing_data_23to1.json", "w") as f:
    # with open("routing_data_5to1.json", "w") as f:
    json.dump(token_data, f, indent=2)
