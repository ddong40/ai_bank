

### pytorch 마스터!! ###


import torch
import tiktoken 
import os 
from torch.utils.data import DataLoader, Dataset 

### 1. Data ### 
tokenizer = tiktoken.get_encoding('gpt2')

text = 'My name is pizza.'

tokens = tokenizer.encode(text)
# print(tokens) [3666, 1438, 318, 14256, 13]

SAVE_DIR = 'C:/Users/jsy/Desktop/coretech/sLLM/sLLM/save/'
os.makedirs(SAVE_DIR, exist_ok=True)


class MyDataSet(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

with open('C:/Users/jsy/Desktop/coretech/sLLM/sLLM/data/cleaned_Harry Potter and the Chamber of Secrets.txt', 'r', encoding='utf-8-sig') as f:
    txt = f.read()

dataset = MyDataSet(txt, max_length= 32, stride = 4)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
print(iter(dataloader))

dataiter = iter(dataloader)

x, y = next(dataiter)

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))

### 2. Model ###

### 모델 ### 

VOCAB_SIZE = tokenizer.n_vocab # 50257 Tiktoken 

CONTEXT_LENGTH = 128 
EMB_DIM = 768 
NUM_HEADS = 12 # Number of attention heads 
NUM_LAYERS = 12 
DROP_RATE = 0.1 
QKV_BIAS = False 

import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        
        assert d_out & NUM_HEADS == 0, "d_out must be divisible by n_heads"
        
        self.d_out = d_out 
        self.head_dim = d_out // NUM_HEADS 
        
        self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(DROP_RATE)
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

    def forward(self, x):
        b, num_token, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_token, NUM_HEADS, self.head_dim)
        values = values.view(b, num_token, NUM_HEADS, self.head_dim)
        queries = queries.view(b, num_token, NUM_HEADS, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:num_token, :num_token] # 모르겠음.
        
        attn_scores.masked_fill_(mask_bool, -torch.inf) 
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_token, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        assert d_out & NUM_HEADS == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.head_dim = d_out // NUM_HEADS
        self.W_query = nn.Linear(d_in, d_out, bias = QKV_BIAS)
        self.W_key = nn.Linear(d_in, d_out, bias = QKV_BIAS)
        self.W_value = nn.Linear(d_in, d_out, bias = QKV_BIAS)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(DROP_RATE)
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1)) # register buffer란 mask라는 버퍼를 등록해주는 것. 이후 self.mask로 불러와서 사용할 수 있음

    def qkv(self, x):
        b, num_token, d_in = x.shape

        key = self.W_key(x)
        query = self.W_query(x)
        value= self.W_value(x) 

        w_key = key.view(b, num_token, NUM_HEADS, self.head_dim)
        w_query = query.view(b, num_token, NUM_HEADS, self.head_dim)
        w_value = value.view(b, num_token, NUM_HEADS, self.head_dim)

        key = w_key.transpose(1,2)
        query = w_query.transpose(1,2)
        value = w_value.transpose(1,2)
        return key, query, value, num_token

    def mask(self, num_token):
        mask_bool = self.mask.bool()[:num_token, :num_token]
        return mask_bool

    def matmul(self, key, query, mask_bool):
        attention = key @ query.transpose(2,3)
        attention = attention.mask_fill_(mask_bool, -torch.inf)
        return attention, key

    def softmax(self, attention, key, value):
        attention = torch.softmax(attention/key[-1]**0.5, dim=-1)
        attention = (attention@value).transpose(1,2)
        context_vec = self.dropout(attention)
        return context_vec

    def projection(self, context_vec, b, num_token):
        context_vec = context_vec.reshape(b, num_token, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

    def forward(self, x):
        key, query, value, num_token = self.qkv(x)
        mask_bool = self.mask(num_token)
        attention, key = self.matmul(key, query, mask_bool)
        context_vec = self.softmax(attention, key, value)
        context_vec = self.projection(context_vec)
        return context_vec


### 3. train / compile ### 


### 4. evaluate / compile ### 