import torch
import numpy as np
import pandas as pd
#print(torch.cuda.is_available())

print("IS CUDA AVAILABE?: "+str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(" ".join(chars))
print(vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}                          # string to integer mapping
itos = {i:ch for i,ch in enumerate(chars)}                          # int to string mapping
encode = lambda string: [stoi[character] for character in string]   # encode = take a string, output list of integers
decode = lambda list: [itos[integer] for integer in list]           # decode = take a list of integers, output string

# Example use
#print(encode("Hehe hello people"))
#print(decode(encode("Hehe hello people")))

data = torch.tensor(encode(text), dtype=torch.long, device=device)
#print(data.shape, data.dtype)
#print(data[:1000])

# val/train set splitting
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 32     # how many independent sequences can we process parallel at a time
block_size = 8      # size of chunks to feed into transformer for training ( max context length for prediction)
eval_iter = 200     # how many evaluation iterations
max_iters = 3000
learning_rate = 1e-2
#print(train_data[:block_size+1])

def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Generating random offset within data, n = batch_size
    ix = torch.randint(len(data)-block_size, (batch_size,))
    # Getting values from dataset based on ix, .stack() stacks them into single tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y is offset by one, so we can use it in training
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

xb,yb = get_batch('train')
print("INPUTS:")
print(xb.shape)
print(xb)
print("TARGETS:")
print(yb.shape)
print(yb)

"""
This tensor (4,8) - 4 rows, 8 columnns contains 32 examples for transformer to use.
"""
def print_examples(xb, yb, batch_size = 4, block_size = 8):
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"When input is {context.tolist()} the target is {target}")

# print_examples(xb, yb)

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) - batch time channel tensor ( B = 4, T = 8, C = vocab_size -- 65)
        if targets is None:                      # if targets not provided dont do anything, return logits later
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # stretching array to 2 dimensions, so it fits into our cross_entropy function(channel needs to come second)
            targets = targets.view(B*T)
            # loss - Cross entropy measures quality of logits with respect to targets. We know next character, how well are we doing it?
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices here
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)            #when inheriting nn.Module, calling self runs forward function basically.
            # focus on last time stem
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)
        return idx



m = BigramLanguageModel(vocab_size)
m = m.to(device)

logits, loss = m(xb,yb)
print(logits.shape)
print(loss)


idx = torch.zeros((1,1), dtype=torch.long, device=device)
# Non trained model, outpust gibberish
print("".join(decode(m.generate(idx, max_new_tokens = 300)[0].tolist())))

# create torch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=2e-21)

training_steps = 15000
for steps in range(training_steps):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)   # zeroing gradients from previous step
    loss.backward()                         # gradients for all parameters
    optimizer.step()                        # update params
    if steps%(training_steps//10) == 0:
        print("Done "+str(100*(steps/training_steps))+"% so far.")
print("loss:" + str(loss.item()))
print("\n\n post-optimization gen:")
print("".join(decode(m.generate(idx, max_new_tokens = 300)[0].tolist())))
