import torch
import numpy as np
import pandas as pd
#print(torch.cuda.is_available())


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

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

# val/train set splitting
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4      # how many independent sequences can we process parallel at a time
block_size = 8      # size of chunks to feed into transformer for training ( max context length for prediction)
print(train_data[:block_size+1])

def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data




