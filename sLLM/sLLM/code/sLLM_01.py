import tiktoken # pip install tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

text = "Harry Potter was a wizard."

tokens = tokenizer.encode(text)

import torch  
from torch.utils.data import Dataset, DataLoader 


class MyDataSet(Dataset):
    def __init__(self, txt, max_lenth, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt) 
        print(token_ids)
        
        print('# of tokens in txt:', len(token_ids))
        
        for i in range(0, len(token_ids) - max_lenth, stride):
            input_chunk = token_ids[i:i + max_lenth]
            target_chunk = token_ids[i + 1 : i + max_lenth + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
with open("jeon/study/LLM/data/cleaned_Harry Potter and the Chamber of Secrets.txt", 'r', encoding='utf-8-sig') as f:
    txt = f.read()

# of tokens in txt: 130520

dataset = MyDataSet(txt, max_lenth = 32, stride = 4)

train_loader = DataLoader(dataset, batch_size = 128, shuffle = True, drop_last = True)

dataiter = iter(train_loader)

x, y = next(dataiter)

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))

# �d have learned by now just to stay put.” Soon, the crowd of gnomes in the field started walking away in a straggling line
# d have learned by now just to stay put.” Soon, the crowd of gnomes in the field started walking away in a straggling line,

## 모델 

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
        
        mask_bool = self.mask.bool()[:num_token, :num_token] 
        
        attn_scores.masked_fill(mask_bool, -torch.inf) # 무한대 
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2) 
        context_vec = context_vec.reshape(b, num_token, self.d_out)  
        context_vec = self.out_proj(context_vec)
        
        return context_vec 
    
     
        
        