"""
Power Attention 기반 MNIST 손글씨 숫자 분류 모델
- 데이터셋: MNIST (28x28 픽셀 손글씨 숫자 이미지)
- 모델: Power Attention (Retention Mechanism 기반)
- 목적: 0-9 숫자 분류 (10개 클래스)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import math

# 스크립트 파일 기준 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULT_DIR = 'C:/Users/jsy/Desktop/coretech/study/mnist/result'

# 결과 디렉토리 생성 (없으면 생성)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. Power Attention의 핵심: Symmetric Power (SPOW) 함수 및 Recurrent State 업데이트
# SPOW 함수는 Lemma 4.2를 기반으로 합니다.
# 실제 구현에서는 효율성을 위해 Tiled SPOW (TSPOW)를 사용하지만,
# 여기서는 개념을 보여주기 위해 SPOW를 직접 구현합니다.
# 높은 p 값에 대해서는 차원이 너무 커져서 실제 GPU에서 비효율적입니다.
# d는 head_dim, p는 power degree입니다.

def spow_transform(x, p):
    """
    Symmetric Power (SPOW) transformation as defined in Lemma 4.2.
    벡터화된 구현으로 성능 최적화.
    This generates a D-dimensional vector from a d-dimensional vector.
    D = binomial(d + p - 1, p)
    
    Args:
        x: (..., d) 형태의 텐서
        p: power degree
    
    Returns:
        (..., D) 형태의 텐서
    """
    d = x.shape[-1]
    
    if p == 1:
        return x
    
    if p == 2:
        # 완전 벡터화된 구현: p=2인 경우
        # SPOW_2(x) = [x_1^2, sqrt(2)x_1x_2, x_2^2, ... sqrt(2)x_d-1x_d, x_d^2]
        output_dim = d * (d + 1) // 2
        
        # x_i^2 항들: (..., d)
        squares = x * x  # (..., d)
        
        # sqrt(2) * x_i * x_j 항들 (i < j) - 완전 벡터화
        # 외적을 사용하여 모든 x_i * x_j 계산: (..., d, d)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)  # (..., d, d)
        
        # 상삼각 행렬의 비대각선 원소들만 추출 (i < j)
        # torch.triu를 사용하여 상삼각 행렬 추출 후 비대각선 원소만 선택
        sqrt2 = math.sqrt(2.0)
        triu_mask = torch.triu(torch.ones(d, d, dtype=torch.bool, device=x.device), diagonal=1)
        cross_terms = sqrt2 * outer[..., triu_mask]  # (..., d*(d-1)//2)
        
        # 결과 결합: [squares, cross_terms]
        result = torch.cat([squares, cross_terms], dim=-1)  # (..., output_dim)
        
        return result
    else:
        raise NotImplementedError("SPOW for p > 2 is not implemented in this conceptual example for simplicity.")


class PowerAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, p, gamma, dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.p = p # Power degree
        self.gamma = gamma # Gating factor (for recurrent state decay)

        # Q, K, V 프로젝션
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        # SPOW 변환 후의 차원 D 계산 (Lemma 4.2)
        # d는 head_dim
        if p == 1:
            self.spow_dim = self.head_dim
        elif p == 2:
            self.spow_dim = self.head_dim * (self.head_dim + 1) // 2
        else:
            raise NotImplementedError("SPOW for p > 2 not implemented for simplicity.")
        
        # Gating parameter (learned per head, or shared)
        # 논문에서는 timestep마다 학습되는 gating value를 제안 (Lin et al. 2025)
        # 여기서는 단순화를 위해 head-wise gamma를 가정 (또는 학습 가능하게)
        # self.gating_weight = nn.Parameter(torch.ones(1)) # 학습 가능한 스칼라 감마 또는 num_heads
        # 논문에서는 Recurrent Mode에서 Si+1 = gamma * Si + ... 형태를 사용합니다.
        # Chunked form에서는 cumulative gated sum of states를 사용합니다.
        # 여기서는 Recurrent Mode에 가까운 형태로 구현합니다.
        
    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        device = x.device

        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Recurrent mode of Power Attention (최적화된 구현)
        # S_0 = 0 (D x v_head_dim)
        # S_i = gamma * S_{i-1} + SPOW_p(K_i)^T V_i
        # Y_i = SPOW_p(Q_i) S_i
        
        # 메모리 효율적인 구현: Chunk 단위로 처리하여 메모리 사용량 감소
        # S is the state: (batch_size, num_heads, spow_dim, head_dim)
        state = torch.zeros(batch_size, self.num_heads, self.spow_dim, self.head_dim, dtype=x.dtype, device=device)
        outputs = []
        
        # Chunk 단위로 처리 (속도 향상을 위해 더 큰 chunk 사용)
        chunk_size = min(128, sequence_length)  # 한 번에 처리할 시퀀스 길이 (더 크게 설정하여 반복 횟수 감소)
        
        for chunk_idx, chunk_start in enumerate(range(0, sequence_length, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, sequence_length)
            chunk_len = chunk_end - chunk_start
            
            # Chunk 단위로 Q, K, V 추출
            q_chunk = q[:, chunk_start:chunk_end]  # (B, chunk_len, H, head_dim)
            k_chunk = k[:, chunk_start:chunk_end]  # (B, chunk_len, H, head_dim)
            v_chunk = v[:, chunk_start:chunk_end]  # (B, chunk_len, H, head_dim)
            
            # SPOW 변환을 chunk 단위로 수행
            # contiguous()를 호출하여 메모리 레이아웃을 보장한 후 reshape
            q_chunk_flat = q_chunk.contiguous().reshape(-1, self.head_dim)  # (B*chunk_len*H, head_dim)
            k_chunk_flat = k_chunk.contiguous().reshape(-1, self.head_dim)  # (B*chunk_len*H, head_dim)
            
            # SPOW 변환 수행 (메모리 사용량이 크므로 즉시 처리)
            spow_k_chunk_flat = spow_transform(k_chunk_flat, self.p)  # (B*chunk_len*H, spow_dim)
            spow_q_chunk_flat = spow_transform(q_chunk_flat, self.p)  # (B*chunk_len*H, spow_dim)
            
            # 원래 shape로 복원
            spow_k_chunk = spow_k_chunk_flat.reshape(batch_size, chunk_len, self.num_heads, self.spow_dim)
            spow_q_chunk = spow_q_chunk_flat.reshape(batch_size, chunk_len, self.num_heads, self.spow_dim)
            
            # 중간 텐서 메모리 해제 (속도 향상을 위해 최소화)
            del q_chunk_flat, k_chunk_flat, spow_k_chunk_flat, spow_q_chunk_flat
            
            # Chunk 내에서 Recurrent 처리
            for i in range(chunk_len):
                k_i = k_chunk[:, i]  # (batch_size, num_heads, head_dim)
                v_i = v_chunk[:, i]  # (batch_size, num_heads, head_dim)
                spow_k_i = spow_k_chunk[:, i]  # (batch_size, num_heads, spow_dim)
                spow_q_i = spow_q_chunk[:, i]  # (batch_size, num_heads, spow_dim)
                
                # Update state: S_i = gamma * S_{i-1} + SPOW_p(K_i)^T V_i
                current_kv_contribution = torch.matmul(v_i.unsqueeze(-1), spow_k_i.unsqueeze(-2)).transpose(-2, -1)
                state = self.gamma * state + current_kv_contribution
                
                # Query state to get output: Y_i = SPOW_p(Q_i) S_i
                output_i_per_head = torch.matmul(spow_q_i.unsqueeze(-2), state).squeeze(-2)
                outputs.append(output_i_per_head)
            
            # Chunk 처리 후 메모리 정리 (속도 향상을 위해 최소화)
            del q_chunk, k_chunk, v_chunk, spow_k_chunk, spow_q_chunk
        
        # Concatenate outputs from all timesteps
        output = torch.stack(outputs, dim=1)  # (batch_size, sequence_length, num_heads, head_dim)
        del outputs

        # 헤드들을 다시 결합
        output = output.contiguous().view(batch_size, sequence_length, hidden_size)
        output = self.output_proj(output)
        output = self.dropout(output)
        return output

class PowerAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, p, gamma, dropout_rate=0.0):
        super().__init__()
        self.power_attention = PowerAttentionLayer(hidden_size, num_heads, p, gamma, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Power Attention + Add & Norm
        x = x + self.dropout(self.power_attention(self.norm1(x)))
        # MLP + Add & Norm
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# 2. Power Attention 기반 MNIST 분류기 아키텍처
class PowerAttentionClassifier(nn.Module):
    def __init__(self, num_classes, img_size, hidden_size, num_layers, num_heads, p, gamma, dropout_rate=0.0):
        super().__init__()
        self.img_flatten_dim = img_size * img_size
        
        # 각 픽셀 (스칼라 값)을 hidden_size 차원으로 임베딩
        # (batch_size, 784, 1) -> (batch_size, 784, hidden_size)
        self.pixel_embedding = nn.Linear(1, hidden_size)
        
        # 학습 가능한 위치 임베딩 (시퀀스 모델링에 필수)
        # Power Attention 자체는 절대 위치 정보를 직접 처리하지 않으므로,
        # 외부에서 위치 정보를 주입해야 합니다.
        self.position_embedding = nn.Parameter(torch.randn(1, self.img_flatten_dim, hidden_size))

        self.layers = nn.ModuleList([
            PowerAttentionBlock(hidden_size, num_heads, p, gamma, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size) # 최종 LayerNorm
        self.classifier = nn.Linear(hidden_size, num_classes) # 분류 헤드

    def forward(self, x):
        # x: (batch_size, 1, H, W)
        batch_size = x.shape[0]
        
        # Flatten image to pixel sequence: (batch_size, 784)
        x = x.view(batch_size, self.img_flatten_dim)
        
        # Each pixel value (scalar) needs to be embedded into hidden_size.
        # (batch_size, 784) -> (batch_size, 784, 1)
        x = x.unsqueeze(-1) 
        
        # Embed each pixel into hidden_size
        # (batch_size, 784, 1) -> (batch_size, 784, hidden_size)
        x = self.pixel_embedding(x)
        
        # Add position embedding
        x = x + self.position_embedding
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Classify based on the average of all pixel embeddings
        x = x.mean(dim=1) # (batch_size, hidden_size)
        logits = self.classifier(x)
        return logits


# 3. MNIST 데이터 로딩 및 전처리 (이전과 동일)
transform = transforms.Compose([
    transforms.ToTensor(), # 이미지를 텐서로 변환 (0~1 범위)
    transforms.Normalize((0.1307,), (0.3081,)), # 평균과 표준편차로 정규화
])

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

# 훈련 속도 향상을 위해 배치 크기 증가 (메모리 허용 범위 내)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# 4. 학습 루프 설정 및 실행 (이전과 동일)
def train_model(model, train_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # 학습 진행 상황 표시
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                          ncols=100, leave=False)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Progress bar 업데이트 (손실, 학습률 표시)
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
            
            # 메모리 정리 (속도 향상을 위해 빈도 감소)
            del output, loss
            if torch.cuda.is_available():
                if (batch_idx + 1) % 20 == 0:  # 20배치마다 메모리 정리 (빈도 감소)
                    torch.cuda.empty_cache()
        
        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        
        # 평가
        model.eval()
        test_loss = 0
        correct = 0
        
        # 테스트 진행 상황 표시
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]', 
                        ncols=100, leave=False)
        
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Progress bar 업데이트
                current_acc = 100. * correct / ((test_pbar.n + 1) * test_loader.batch_size)
                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
                # 메모리 정리
                del output, pred, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        test_pbar.close()
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        # 에포크 결과 출력
        print(f'\nEpoch: {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%')
        print('-' * 80)


def visualize_predictions(model, test_loader, device, num_samples=16, save_path=None):
    """
    예측 결과 시각화
    - 테스트 이미지와 예측 결과를 함께 표시
    - 정답과 예측이 다른 경우를 우선적으로 표시
    """
    model.eval()
    
    # 그리드 크기 계산 (4x4 또는 2x8)
    if num_samples <= 16:
        rows, cols = 4, 4
    else:
        rows, cols = int(np.sqrt(num_samples)), int(np.ceil(num_samples / int(np.sqrt(num_samples))))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            for i in range(len(data)):
                img = data[i].cpu().squeeze()
                true_label = target[i].item()
                pred_label = predicted[i].item()
                
                if true_label == pred_label:
                    correct_samples.append((img, true_label, pred_label))
                else:
                    incorrect_samples.append((img, true_label, pred_label))
                
                # 충분한 샘플을 수집했으면 종료
                if len(correct_samples) + len(incorrect_samples) >= num_samples * 2:
                    break
            
            if len(correct_samples) + len(incorrect_samples) >= num_samples * 2:
                break
    
    # 잘못 예측한 샘플을 먼저 표시하고, 나머지는 정답 샘플로 채움
    display_samples = incorrect_samples[:num_samples//2] + correct_samples[:num_samples - len(incorrect_samples[:num_samples//2])]
    
    for idx, (img, true_label, pred_label) in enumerate(display_samples[:num_samples]):
        axes[idx].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(f'True: {true_label}, Pred: {pred_label}', color=color, fontsize=10)
        axes[idx].axis('off')
    
    # 사용하지 않는 subplot 숨기기
    for idx in range(len(display_samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'mnist_predictions_retention.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n예측 결과가 '{save_path}'에 저장되었습니다.")
    plt.close()

# 모델 파라미터 설정 (훈련 속도 향상을 위해 경량화)
num_classes = 10
img_size = 28
hidden_size = 64   # 임베딩 차원 (더 작게 설정하여 속도 향상)
num_layers = 1     # Power Attention 블록 개수 (2 -> 1로 감소)
num_heads = 4      # 멀티 헤드 어텐션 헤드 개수
p = 2              # Power Attention의 p (논문에서 p=2는 좋은 성능을 보임)
gamma = 0.9        # Recurrent State의 decay factor (RetNet과 유사하게 사용)
dropout_rate = 0.1
epochs = 100    # 에포크 수 감소 (10 -> 5)
lr = 2e-3          # 학습률 증가 (더 빠른 수렴)

# GPU 전용 실행
if not torch.cuda.is_available():
    raise RuntimeError("CUDA를 사용할 수 없습니다. GPU가 필요합니다.")

device = torch.device("cuda")

# GPU 메모리 정리 및 확인
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
free = total_memory - reserved

print(f"GPU 메모리: {total_memory:.2f} GB")
print(f"할당된 메모리: {allocated:.2f} GB")
print(f"예약된 메모리: {reserved:.2f} GB")
print(f"사용 가능한 메모리: {free:.2f} GB")

if free < 1.0:  # 1GB 미만이면 경고
    print("⚠️  경고: GPU 메모리가 부족합니다. 다른 프로세스를 종료하거나 배치 크기를 더 줄이세요.")

model = PowerAttentionClassifier(num_classes, img_size, hidden_size, num_layers, num_heads, p, gamma, dropout_rate)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"사용 디바이스: {device}")

# 모델 훈련
train_model(model, train_loader, test_loader, epochs, lr, device)

# 모델 저장
model_path = os.path.join(RESULT_DIR, 'mnist_model_retention.pth')
torch.save(model.state_dict(), model_path)
print(f"\n모델이 '{model_path}'에 저장되었습니다.")

# 예측 결과 시각화
print("\n예측 결과 시각화 중...")
visualize_predictions(model, test_loader, device, num_samples=16)

