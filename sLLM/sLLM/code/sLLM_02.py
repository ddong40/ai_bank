import tiktoken # pip install tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

text = "Harry Potter was a wizard."

tokens = tokenizer.encode(text)

import torch  
from torch.utils.data import Dataset, DataLoader 
import os

# 모델 저장 경로 설정
SAVE_DIR = "/home/ct/cylinder/jeon/study/LLM/save/sLLM02/"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    
with open("/home/ct/cylinder/jeon/study/LLM/data/cleaned_Harry Potter and the Chamber of Secrets.txt", 'r', encoding='utf-8-sig') as f:
    txt = f.read()

# of tokens in txt: 130520

dataset = MyDataSet(txt, max_lenth = 32, stride = 4)

train_loader = DataLoader(dataset, batch_size = 128, shuffle = True, drop_last = True)

dataiter = iter(train_loader)

x, y = next(dataiter)

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))

#d have learned by now just to stay put." Soon, the crowd of gnomes in the field started walking away in a straggling line
# d have learned by now just to stay put." Soon, the crowd of gnomes in the field started walking away in a straggling line,

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
        
        mask_bool = self.mask.bool()[:num_token, :num_token] # 모르겠음.
        
        attn_scores.masked_fill_(mask_bool, -torch.inf) 
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
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
        var = x.var(dim=-1, keepdim=True, unbiased=False) # layer norm 에선 keepdim이 필수 
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift 
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))
        ))

class FeedForward(nn.Module): 
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMB_DIM, 4 * EMB_DIM),
            GELU(),
            nn.Linear(4 * EMB_DIM, EMB_DIM)
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=EMB_DIM,
            d_out=EMB_DIM
        )
        
        self.ff = FeedForward()
        self.norm1 = LayerNorm(EMB_DIM)
        self.norm2 = LayerNorm(EMB_DIM)
        self.drop_shortcut = nn.Dropout(DROP_RATE)
        
    def forward(self, x):
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x 
        
        return x 
    
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
        self.drop_emb = nn.Dropout(DROP_RATE) 
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = LayerNorm(EMB_DIM) 
        self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False) 
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x) 
        logits = self.out_head(x) 
        
        return logits  

### 3. 훈련 ### 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device) 
# cuda 

torch.manual_seed(123)

model = GPTModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

token_seen, global_step = 0, -1

# losses = []

# for epoch in range(100):
#     model.train() # Set model to training mode 
    
#     epoch_loss = 0 
#     for input_batch, target_batch in train_loader:
#         optimizer.zero_grad() # Reset loss gradients from previous batch iteration 
#         input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        
#         logits = model(input_batch) 
#         loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#         epoch_loss += loss.item() 
#         loss.backward() # Calculate loss gradients 
#         optimizer.step() # Update model weights using loss gradients
#         token_seen += input_batch.numel()
#         global_step += 1
        
#         if global_step % 1000 ==0:
#             print(f'Tokens seen: {token_seen}')
#         #Optional evaluation step 
    
#     # 에포크가 끝난 후에만 실행
#     avg_loss = epoch_loss / len(train_loader)
#     losses.append(avg_loss)
#     print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
    
#     # 모델 저장 (에포크마다 한 번만)
#     model_filename = f"model_{str(epoch + 1).zfill(3)}.pth"
#     model_path = os.path.join('/home/ct/cylinder/jeon/study/LLM/save/sLLM02', model_filename)
#     torch.save(model.state_dict(), model_path)
#     print(f"모델 저장됨: {model_path}")

# import matplotlib.pyplot as plt  

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.title('TrainingLoss over Epochs')
# plt.show()

### 4. 평가 예측 ### 

# 모델 로드 시에도 경로 변수 사용
model_path = os.path.join(SAVE_DIR, "model_100.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval() # dropout을 사용하지 않음 

idx = tokenizer.encode("Dobby is") 
idx = torch.tensor(idx).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(idx)

logits = logits[:, -1, :] # 전체 배치 중에서 마지막 행의 전체 토큰에 대하여 슬라이싱 

# 가장 높은 확률 단어 10개 출력 
top_logits, top_indices = torch.topk(logits, 10)
for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
    print(f'{p:.2f}\t {i}\t {tokenizer.decode([i])}') # f 구문 다시 공부하기 
    
# 가장 확률이 높은 단어 출력
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
flat = idx_next.squeeze(0) # 배치 차원 제거 
out = tokenizer.decode(flat.tolist()) 

print(out)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k) 
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits) 
            
        if temperature > 0.0:
            logits = logits / temperature 
            probs = torch.softmax(logits, dim = -1) # (batch_size, context_len )
            idx_next = torch.multinomial(probs, num_samples=1) # batch_size ,1 
            
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (batch_size, 1)
            
        if idx_next == eos_id:
            break 
        
        idx = torch.cat((idx, idx_next), dim=1) # (batch_size, num_token+1)
    
    return idx  

start_context = input("Start context:") 

# idx = tokenizer.encode(start_context, allowed_special={'<>})) 

idx = tokenizer.encode(start_context)
idx = torch.tensor(idx).unsqueeze(0)

context_size = model.pos_emb.weight.shape[0]

for i in range(10):
    
    token_ids = generate(model= model, idx=idx.to(device), max_new_tokens=50, context_size=context_size,
                         top_k=50, temperature=0.5)
    
    flat = token_ids.squeeze(0) # remove batch dimension 
    out = tokenizer.decode(flat.tolist()).replace("\n", "")
    
    print(i, ":", out) 
    
