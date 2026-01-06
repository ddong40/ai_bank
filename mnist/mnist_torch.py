"""
MNIST 손글씨 숫자 분류를 위한 PyTorch 구현
- 데이터셋: MNIST (28x28 픽셀 손글씨 숫자 이미지)
- 모델: Convolutional Neural Network (CNN)
- 목적: 0-9 숫자 분류 (10개 클래스)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 스크립트 파일 기준 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')

# 결과 디렉토리 생성 (없으면 생성)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 하이퍼파라미터 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용 중인 디바이스: {DEVICE}")


class MNIST_CNN(nn.Module):
    """
    MNIST 분류를 위한 CNN 모델
    - 입력: 28x28 그레이스케일 이미지
    - 출력: 10개 클래스에 대한 확률 분포
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 두 번째 컨볼루션 블록
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 드롭아웃 레이어
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 완전 연결 레이어
        # 28x28 -> 14x14 (pool1) -> 7x7 (pool2)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # 첫 번째 컨볼루션 블록
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 두 번째 컨볼루션 블록
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # 완전 연결 레이어
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def load_data():
    """
    MNIST 데이터셋 로딩 및 전처리
    - 훈련 데이터와 테스트 데이터 분리
    - 정규화 및 텐서 변환
    """
    # 데이터 전처리: 정규화 (평균 0.1307, 표준편차 0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 훈련 데이터셋
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    # 테스트 데이터셋
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device):
    """
    모델 학습 함수
    - 한 에포크 동안 학습 수행
    - 손실 및 정확도 계산
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 순전파
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계 계산
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 진행 상황 출력 (100 배치마다)
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    모델 평가 함수
    - 테스트 데이터셋에 대한 성능 평가
    - 손실 및 정확도 계산
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc


def visualize_predictions(model, test_loader, device, num_samples=8):
    """
    예측 결과 시각화
    - 테스트 이미지와 예측 결과를 함께 표시
    """
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            for i in range(min(num_samples, len(data))):
                img = data[i].cpu().squeeze()
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'True: {target[i].item()}, Pred: {predicted[i].item()}')
                axes[i].axis('off')
            
            break
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'mnist_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"예측 결과가 '{save_path}'에 저장되었습니다.")
    plt.close()


def main():
    """
    메인 실행 함수
    - 데이터 로딩
    - 모델 초기화
    - 학습 및 평가 루프
    """
    print("=" * 50)
    print("MNIST 손글씨 숫자 분류 학습 시작")
    print("=" * 50)
    
    # 데이터 로딩
    print("\n[1/4] 데이터 로딩 중...")
    train_loader, test_loader = load_data()
    print(f"훈련 데이터: {len(train_loader.dataset)}개")
    print(f"테스트 데이터: {len(test_loader.dataset)}개")
    
    # 모델 초기화
    print("\n[2/4] 모델 초기화 중...")
    model = MNIST_CNN().to(DEVICE)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 및 평가 루프
    print("\n[3/4] 학습 시작...")
    print(f"에포크 수: {NUM_EPOCHS}, 배치 크기: {BATCH_SIZE}, 학습률: {LEARNING_RATE}")
    print("-" * 50)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'\n에포크 [{epoch}/{NUM_EPOCHS}]')
        print('-' * 30)
        
        # 학습
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 평가
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'\n에포크 [{epoch}/{NUM_EPOCHS}] 요약:')
        print(f'  훈련 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'  테스트 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%")
    
    # 학습 곡선 시각화
    print("\n[4/4] 학습 곡선 시각화 중...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 손실 곡선
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 정확도 곡선
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Curves - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'mnist_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"학습 곡선이 '{save_path}'에 저장되었습니다.")
    plt.close()
    
    # 예측 결과 시각화
    visualize_predictions(model, test_loader, DEVICE)
    
    # 모델 저장
    model_path = os.path.join(RESULT_DIR, 'mnist_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"모델이 '{model_path}'에 저장되었습니다.")


if __name__ == '__main__':
    main()

