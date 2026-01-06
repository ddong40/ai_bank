"""
sLLM 파인튜닝 스크립트 - Validation 포함 버전
- Midm-2.0-Mini-Instruct 모델을 사용한 파인튜닝
- 커스텀 Q&A 데이터로 학습
- Train/Validation 분할 및 성능 평가 기능 포함
- Early stopping 및 best model 저장 기능 포함
- 학습 과정 시각화 기능 포함
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

model_name = "K-intelligence/Midm-2.0-Mini-Instruct"

save_path = "C:/Users/jsy/Desktop/coretech/sLLM/sLLM/save/sLLM_finetuning_instruct_validation_01"

# 저장 경로 생성
os.makedirs(save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # <|end_of_text|> 128001

qna_list = []
with open("C:/Users/jsy/Desktop/coretech/sLLM/sLLM/data/jeoncustomdata.txt", "r", encoding="utf-8") as file:
    for line in file:
        qna = line.strip().split('|') # 안내: 입력 문서의 '|'는 질문과 답변을 구분하는 문자
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."}, # 모든 질문 공통
            {"role": "user", "content": qna[0]},     # 질문 부분
            {"role": "assistant", "content": qna[1]} # 답변 부분
        ]
        q = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        input_str = tokenizer.apply_chat_template(messages[:3], tokenize=False, add_generation_prompt=True)
        # print(input_str)
        input_str = input_str[:-len('start_header_id\>assistant<|end_header_id|>')-4]
        # print(input_str)
        # print("--------------")
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        input_ids = tokenizer.encode(input_str, add_special_tokens=False)
        qna_list.append({'q':q, 'input':input_str, 'q_ids':q_ids, 'input_ids':input_ids})

max_length = max(len(i['input_ids']) for i in qna_list)

print(f"전체 데이터셋 크기: {len(qna_list)}")
print(f"최대 토큰 길이: {max_length}")

# 데이터셋을 train/validation으로 분할 (8:2 비율)
train_qna, val_qna = train_test_split(qna_list, test_size=0.2, random_state=42)
print(f"훈련 데이터: {len(train_qna)}개")
print(f"검증 데이터: {len(val_qna)}개")

# qna_list = []
# with open("C:/Users/jsy/Desktop/coretech/sLLM/sLLM/data/jeoncustomdata.txt", "r", encoding="utf-8") as file:
#     for line in file:
#         qna = line.strip().split('|') # 안내: 입력 문서의 '|'는 질문과 답변을 구분하는 문자
#         input_str = qna[0] + " " + qna[1]
#         item = {'q':qna[0], 'input':input_str, 'q_ids':tokenizer.encode(qna[0]), 'input_ids':tokenizer.encode(input_str)}
#         qna_list.append(item)

# max_length = max(len(item['input_ids']) for item in qna_list) # + 1은 질문답변 사이의 빈칸

# print(qna_list)
# print(max_length)


import torch
from torch.utils.data import Dataset, DataLoader

EOT = 128009 # instruct 모델과 다름

class MyDataset(Dataset):
    def __init__(self, qna_list, max_length):
        self.input_ids = []
        self.target_ids = []

        for qa in qna_list:
            token_ids = qa['input_ids']
            input_chunk = token_ids
            target_chunk = token_ids[1:]
            input_chunk += [EOT]* (max_length - len(input_chunk))
            target_chunk +=  [EOT]* (max_length - len(target_chunk))
            len_ignore = len(qa['q_ids']) - 1 # target은 한 글자가 짧기 때문
            target_chunk[:len_ignore] = [-100] * len_ignore

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 훈련 및 검증 데이터셋 생성
train_dataset = MyDataset(train_qna, max_length=max_length)
val_dataset = MyDataset(val_qna, max_length=max_length)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
i = iter(train_loader)
x, y = next(i)

y_temp = y[0].tolist()
y_temp = [x for x in y_temp if x != -100] # -100은 제외하고 디코딩

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y_temp))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
torch.manual_seed(123)
model.to(device)

# Validation 평가 함수
def evaluate_model(model, val_loader, device):
    """모델의 validation loss를 계산하는 함수"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            logits = model(input_batch).logits
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

# Early stopping을 위한 클래스
class EarlyStopping:
    """Early stopping을 위한 클래스"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """현재 최고 성능 모델의 상태 저장"""
        self.best_weights = model.state_dict().copy()

# 파인튜닝 전에 어떻게 대답하는지 확인
print("\n=== 파인튜닝 전 모델 응답 ===")
questions = [qna['q_ids'] for qna in train_qna[:3]]  # 처음 3개만 테스트

for i, q_ids in enumerate(questions):
    model.eval()
    with torch.no_grad():
        output = model.generate(
            torch.tensor([q_ids]).to("cuda"),
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# 초기 validation loss 계산
initial_val_loss = evaluate_model(model, val_loader, device)
print(f"\n파인튜닝 전 Validation Loss: {initial_val_loss:.4f}")

tokens_seen, global_step = 0, -1

# 학습 과정 추적을 위한 리스트
train_losses = []
val_losses = []
best_val_loss = float('inf')

# Early stopping 및 optimizer 설정
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)


print("\n=== 훈련 시작 ===")
for epoch in range(50):
    # Training phase
    model.train()
    epoch_train_loss = 0
    num_batches = 0
    
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        logits = model(input_batch).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        
        epoch_train_loss += loss.item()
        num_batches += 1
        
        loss.backward()
        optimizer.step()
        tokens_seen += input_batch.numel()
        global_step += 1

    # 평균 훈련 loss 계산
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    val_loss = evaluate_model(model, val_loader, device)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1:2d}/{50} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Best model 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_file = os.path.join(save_path, "best_model.pth")
        torch.save(model.state_dict(), best_model_file)
        print(f"  새로운 최고 성능! Val Loss: {val_loss:.4f} - 모델 저장: {best_model_file}")
    
    # 정기적으로 모델 저장 (매 10 에포크)
    if (epoch + 1) % 10 == 0:
        model_file = os.path.join(save_path, f"model_epoch_{str(epoch).zfill(3)}.pth")
        torch.save(model.state_dict(), model_file)
        print(f"  체크포인트 저장: {model_file}")
    
    # Early stopping 체크
    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch+1}")
        break

# 학습 과정 시각화
print("\n=== 학습 과정 시각화 ===")
plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss 차이 그래프
plt.subplot(1, 2, 2)
loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
plt.plot(epochs_range, loss_diff, 'g-', label='|Train - Val| Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss Difference')
plt.title('Training-Validation Loss Difference')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()

# 최종 결과 출력
print(f"\n=== 최종 결과 ===")
print(f"최고 Validation Loss: {min(val_losses):.4f}")
print(f"최종 Training Loss: {train_losses[-1]:.4f}")
print(f"최종 Validation Loss: {val_losses[-1]:.4f}")
print(f"총 학습 에포크: {len(train_losses)}")

# 오버피팅 체크
if len(val_losses) > 5:
    last_5_val = val_losses[-5:]
    if all(val_losses[-1] >= val for val in last_5_val[:-1]):
        print("⚠️  경고: 최근 5 에포크 동안 validation loss가 개선되지 않았습니다. 오버피팅 가능성이 있습니다.")
    else:
        print("✅ 양호: Validation loss가 적절히 개선되고 있습니다.")

# 파인튜닝 후 best 모델로 테스트
print("\n=== 파인튜닝 후 모델 응답 (Best Model) ===")
best_model_path = os.path.join(save_path, "best_model.pth")
if os.path.exists(best_model_path):
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"Best 모델 로드 성공: {best_model_path}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
    except RuntimeError as e:
        print(f"Best 모델 로드 오류: {e}")
        print("현재 메모리에 있는 학습된 모델을 사용합니다.")
else:
    print(f"Best 모델 파일이 없습니다: {best_model_path}")
    print("현재 메모리에 있는 학습된 모델을 사용합니다.")
model.eval()

# Validation 데이터로 모델 성능 테스트
questions = [qna['q_ids'] for qna in val_qna]  # validation 데이터 사용

for i, q_ids in enumerate(questions):
    model.eval()
    with torch.no_grad():
        output = model.generate(
            torch.tensor([q_ids]).to('cuda'),
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    print(f"Validation Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# 최종 validation loss 재계산
final_val_loss = evaluate_model(model, val_loader, device)
print(f"\n최종 Best Model Validation Loss: {final_val_loss:.4f}")


print("\n=== 인터랙티브 테스트 ===")
print("종료하려면 'quit' 또는 'exit'를 입력하세요.")

while True:
    user_input = input("\n질문을 입력하세요: ")
    if user_input.lower() in ['quit', 'exit', '종료']:
        print("테스트를 종료합니다.")
        break
        
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."},
        {"role": "user", "content": user_input}
    ]

    model.eval()
    with torch.no_grad():
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) 
        output = model.generate(
            torch.tensor([ids]).to('cuda'),
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # 시스템 메시지와 사용자 입력 부분을 제거하고 응답만 추출
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"AI 응답: {response}")