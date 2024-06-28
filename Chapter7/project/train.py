from model import GPT, GPTConfig
import inspect
import math
import time
import tiktoken
import os
import numpy as np
import sys
import torch.nn.functional as F
import torch

"""
- Training data: 185M tokens
- Batch size: 524288 tokens
- Max steps (one epoch): 185M/524288 = 352 steps to go through the entire dataset
"""

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

#Gradient Accumulation
total_batch_size = 524288 # 524288 ~0.5M tokens, This is the batch size in the paper of GPT3 small model
B = 8 # micro batch size - 4 just for testing
T = 1024  # sequence length - 32 just for testing

# Because we can't fit the entire batch in memory, we split it into micro-batches (Gradient Accumulation)
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
warmup_steps = 100
max_steps = 352 # 352 steps is one epoch

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

optimizer = model.config_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    
    # Validation:
    if step % 50 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):       
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        print(f"Validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, 'a') as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if step > 0 and (step % 150 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            
            torch.save(checkpoint, checkpoint_path)

    # generate from the model
    if ((step > 0 and step % 50 == 0) or last_step):
        model.eval()  
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("import pandas as pd\n")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        tokens = tokens.to(device)

        while tokens.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, xcol), dim=1)

            for i in range(num_return_sequences):
                decoded_tokens = tokens[i, :max_length].tolist()
                decoded = enc.decode(decoded_tokens)
                print('>', decoded)         

    # Traning: 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_acc_steps): # Here we are computing the gradients for an entire batch (0.5M tokens)
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type= device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
            logits, loss = model(x, y)

        loss = loss / grad_acc_steps # Normalize the loss
        loss_accum += loss.detach() 
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    print(f"Step {step}, Loss: {loss_accum.item()}, dt: {dt*1000:.2f}ms ")
    with open(log_file, 'a') as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

# sys.exit(0)