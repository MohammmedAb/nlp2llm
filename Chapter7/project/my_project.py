import inspect
import math
import torch.nn as nn
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import tiktoken
import os
import numpy as np
import sys

"""
Model Info:
- GPT model with 12 layers, 12 heads, 768 hidden size
- Number of parameters: 162M
- Dataset: Bigcode/StarCoder dataset
- Tokenizer: cl100k_base
- Training data: 185M tokens
- Batch size: 512k tokens
"""

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1 # init scale for the weights

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # turn the input embedding to query, key, value vectors
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # project the output back to the embedding size

        # Attributes
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPTConfig:
    block_size: int= 1024 # maximum length of the input sequence
    vocab_size: int = 100277 # size of the vocabulary
    n_layer: int = 12 # Transformer layers
    n_head: int = 12 # Attention heads
    n_embd: int = 768 # Embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2* self.config.n_layer) ** -0.5 # scale initialization by the number of layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        


    def forward(self, idx, target = None):
        B, T = idx.size()
        assert T <= self.config.block_size, 'Cannot forward, model block size is exhausted'

        pos = torch.arange(0, T, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
    
    def config_optimizer(self, weight_decay, learning_rate, device):
        # start with all the parameters that require a gradient
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_parameters = [p for n, p in param_dict.items() if p.dim() >= 2] # emdeddings and metrices in the linear layers
        nondecay_parameters = [p for n, p in param_dict.items() if p.dim() < 2] # biases (one dimentional tensors)
        optim_groups = [
            {'params': decay_parameters, 'weight_decay': weight_decay},
            {'params': nondecay_parameters, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_parameters)
        num_nondecay_params = sum(p.numel() for p in nondecay_parameters)
        print(f"number decayed parameter tensor: {len(decay_parameters)} with {num_decay_params:,} parameters")
        print(f"number non-decayed parameter tensor: {len(nondecay_parameters)} with {num_nondecay_params:,} parameters")

        fused_available =  'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused adam: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T

        assert split in {'train', 'val'}
        data_root = "npy_data"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))

        self.current_position += B*T

        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B*T
        return x, y


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = GPT(GPTConfig())
model = model.to(device)
model_num_of_params = sum(p.numel() for p in model.parameters())
print(f"Model has {model_num_of_params} parameters")
enc = tiktoken.get_encoding('cl100k_base')
# sys.exit(0)

total_batch_size = 524288 # 512k tokens
B = 4 # micro batch size - 4 just for testing
T = 32  # sequence length - 32 just for testing
assert total_batch_size % (B*T) == 0, 'Batch size must be divisible by B*T'
grad_acc_steps = total_batch_size // (B*T)
print(f"Total batch size: {total_batch_size}")
print(f"Calculating gradients every {grad_acc_steps} steps")

train_loader = DataLoader(B=B, T=T, split='train')
val_loader = DataLoader(B=B, T=T, split='val')

use_compile = False
if use_compile:
    model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 # total number of steps in one epoch: 185M // 512k = 361

def get_lr(it):
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) min learning rate 
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.cofig_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item()}")

sys.exit(0)
num_return_sequences = 5
max_length = 30
model.eval()
tokens = enc.encode("Hello i'm a language model,")
tokens = torch.tensor(tokens).unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens
torch.manual_seed(22)

while x.size(1) < max_length:
    
    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[:, -1, :]
        
        probs = F.softmax(next_token_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices ,-1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)
