import torch.nn as nn
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import tiktoken

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

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('code_sample.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('cl100k_base')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))

        self.current_position += B*T

        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = GPT(GPTConfig())
model = model.to(device)

train_loader = DataLoaderLite(4, 32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item()}")

import sys; sys.exit(0)
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
