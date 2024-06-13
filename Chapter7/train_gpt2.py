from dataclasses import dataclass
import torch
import torch as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, config):    
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # fully connected layer
        self.gelu = nn.GELU(approximation='tanh') # activation function
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # fully connected layer
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embd) # layernorm 1
        self.attn = CasualSelfAttention(config) # multi-head self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # feedforward neural network
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPTConfig:
    block_size: int= 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT:

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # transformer block (hidden layers)
            ln_f = nn.LayerNorm(config.n_embd) # layernorm after the last block
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language model head

