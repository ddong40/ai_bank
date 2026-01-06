import torch  
import tiktoken 
from torch.utils.data import Dataset, DataLoader 
import os  

tokenizer = tiktoken.get_encoding('gpt2')

text = 'Harry Potter was a wizard.' 

tokens = tokenizer.encode(text) 

# 모델 저장 경로 설정 
SAVE_DIR = '/home/ct/Desktop/project/jeon/LLM/sLLM/save'
os.makedirs(SAVE_DIR, exist_ok=True) # 해당 경로 생성, 없으면 생성, 있으면 생성 안함. 만약 exist_ok=False이면 에러 발생 

class MyDataSet(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        print(token_ids)
        print('# of tokens in txt:', len(token_ids))
        
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
        print(self.input_ids[0:2])
        print(self.target_ids[0:2])
    
    def __len__(self):
        return len(self.input_ids) 
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
with open('/home/ct/Desktop/project/jeon/LLM/sLLM/data/cleaned_Harry Potter and the Chamber of Secrets.txt', 'r', encoding='utf-8') as f:
    txt = f.read()
    
dataset = MyDataSet(txt, max_length=10, stride=4)

train_loader = DataLoader(dataset, batch_size = 128, shuffle= True, drop_last = True)

dataiter = iter(train_loader) # 데이터 로더를 반복자로 변환

x, y = next(dataiter) # 데이터 로더에서 첫 배치 데이터 반환

print(x.shape, y.shape) 
# torch.Size([128, 10]) torch.Size([128, 10])



print(tokenizer.decode(x[0].tolist()), tokenizer.decode(y[0].tolist()))

### 모델 ### 

VOCAB_SIZE = tokenizer.n_vocab  

CONTEXT_LENGTH = 128 
EMB_DIM = 768 
NUM_HEADS = 12
NUM_LAYERS = 12 
DROP_RATE = 0.1 
QKV_BIAS = False 

import torch.nn as nn  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        
        assert d_out & NUM_HEADS == 0, "d_out_must be disvisible by NUM_HEADS"
        
        self.d_out = d_out
        self.head_dim = d_out // NUM_HEADS 
        
        self.W_query = nn.Linear(d_in, d_out, bias = QKV_BIAS)
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
        queries = queries.view(b, num_token, NUM_HEADS, self.head_dim) 
        values = values.view(b, num_token, NUM_HEADS, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values= values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:num_token, :num_token]
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores /keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_token, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec 
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x -mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
