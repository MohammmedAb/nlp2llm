import tiktoken
import torch
from model import GPT
import torch.nn.functional as F
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    print(f"Model at step: {checkpoint['step']} and validation loss: {checkpoint['val_loss']:.3f} loaded")
    return model


def generate_from_model(model, enc, num_return_sequences, max_length, prefix):
    model.eval()  
    tokens = enc.encode(prefix)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--num_sequences", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)   
    print("Loading the model...")
    model = load_model("log/model_00351.pt")
    enc = tiktoken.get_encoding('cl100k_base')
    args = parser.parse_args()
    
    generate_from_model(model, enc, args.num_sequences, args.max_length, args.prefix)
    