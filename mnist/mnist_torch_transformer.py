"""
Transformer 기반 MNIST 손글씨 숫자 분류 모델
- 데이터셋: MNIST (28x28 픽셀 손글씨 숫자 이미지)
- 모델: Transformer (Multi-Head Self-Attention 기반)
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

# 1. Multi-Head Self-Attention 레이어 구현
class MultiHeadSelfAttention(nn.Module):
    """
    표준 Multi-Head Self-Attention 메커니즘
    - Scaled Dot-Product Attention 사용
    - 여러 헤드로 병렬 처리하여 다양한 표현 학습
    """
    def __init__(self, hidden_size, num_heads, dropout_rate=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size는 num_heads로 나누어떨어져야 합니다."
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5  # Scaled dot-product attention의 스케일링 팩터
        
        # Q, K, V 프로젝션 레이어
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        
        # Q, K, V 계산 및 헤드로 분할
        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중치를 Value에 적용
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 헤드들을 결합
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, hidden_size
        )
        
        # 출력 프로젝션
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        return output


# 2. Transformer Block 구현
class TransformerBlock(nn.Module):
    """
    표준 Transformer Encoder Block
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    def __init__(self, hidden_size, num_heads, dropout_rate=0.0):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Self-Attention + Add & Norm
        x = x + self.dropout(self.attention(self.norm1(x)))
        
        # Feed-Forward + Add & Norm
        x = x + self.mlp(self.norm2(x))
        
        return x


# 3. Transformer 기반 MNIST 분류기 아키텍처
class TransformerClassifier(nn.Module):
    """
    Transformer 기반 MNIST 분류 모델
    - 이미지를 픽셀 시퀀스로 변환
    - 위치 임베딩 추가
    - 여러 Transformer Block을 통과
    - 평균 풀링 후 분류
    """
    def __init__(self, num_classes, img_size, hidden_size, num_layers, num_heads, dropout_rate=0.0):
        super().__init__()
        self.img_flatten_dim = img_size * img_size
        
        # 각 픽셀 (스칼라 값)을 hidden_size 차원으로 임베딩
        # (batch_size, 784, 1) -> (batch_size, 784, hidden_size)
        self.pixel_embedding = nn.Linear(1, hidden_size)
        
        # 학습 가능한 위치 임베딩 (시퀀스 모델링에 필수)
        self.position_embedding = nn.Parameter(torch.randn(1, self.img_flatten_dim, hidden_size))
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)  # 최종 LayerNorm
        self.classifier = nn.Linear(hidden_size, num_classes)  # 분류 헤드
        
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
        
        # Transformer Blocks 통과
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Classify based on the average of all pixel embeddings
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        logits = self.classifier(x)
        
        return logits


# 4. MNIST 데이터 로딩 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환 (0~1 범위)
    transforms.Normalize((0.1307,), (0.3081,)),  # 평균과 표준편차로 정규화
])

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

# 배치 크기 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


# 5. 학습 루프 설정 및 실행
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
            
            # 메모리 정리
            del output, loss
            if torch.cuda.is_available():
                if (batch_idx + 1) % 20 == 0:  # 20배치마다 메모리 정리
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
        save_path = os.path.join(RESULT_DIR, 'mnist_predictions_transformer.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n예측 결과가 '{save_path}'에 저장되었습니다.")
    plt.close()


# 모델 파라미터 설정
num_classes = 10
img_size = 28
hidden_size = 64      # 임베딩 차원
num_layers = 2        # Transformer Block 개수
num_heads = 4         # 멀티 헤드 어텐션 헤드 개수
dropout_rate = 0.1
epochs = 30           # 에포크 수
lr = 2e-3             # 학습률

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

model = TransformerClassifier(num_classes, img_size, hidden_size, num_layers, num_heads, dropout_rate)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"사용 디바이스: {device}")

# 모델 훈련
train_model(model, train_loader, test_loader, epochs, lr, device)

# 모델 저장
model_path = os.path.join(RESULT_DIR, 'mnist_model_transformer.pth')
torch.save(model.state_dict(), model_path)
print(f"\n모델이 '{model_path}'에 저장되었습니다.")

# 예측 결과 시각화
print("\n예측 결과 시각화 중...")
visualize_predictions(model, test_loader, device, num_samples=16)
