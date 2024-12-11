# 가상환경 torch241

import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


# Mask R-CNN 모델을 COCO 데이터셋 가중치로 초기화
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

model.eval()  # 모델을 평가 모드로 설정

# 사용할 장치 설정 (GPU가 사용 가능하다면 GPU로 설정)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
# path = 'C:/Users/ddong40/Desktop/paper/'
# 테스트할 이미지 경로 설정
test_image_path = "C:/Users/ddong40/Desktop/paper/image/david.jpg"  # 여기서 이미지 경로를 넣어주세요

# 이미지 불러오기 및 전처리
image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식이므로 RGB로 변환
image_rgb = image_rgb.astype(np.float32) / 255.0  # 정규화
image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 텐서로 변환

# 예측 수행
with torch.no_grad():
    predictions = model(image_tensor)

# 예측 결과 시각화
pred_boxes = predictions[0]["boxes"].cpu().numpy()
pred_scores = predictions[0]["scores"].cpu().numpy()
pred_masks = predictions[0]["masks"].cpu().numpy()

# 신뢰도 임계값 설정 (0.5 이상만 표시)
threshold = 0.5
sample = (image_rgb * 255).astype(np.uint8)  # 원본 이미지로 변환
for i, box in enumerate(pred_boxes):
    if pred_scores[i] >= threshold:
        # 바운딩 박스 그리기
        box = box.astype(int)
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        
        # 마스크 그리기
        mask = pred_masks[i, 0] > threshold
        colored_mask = np.zeros_like(sample, dtype=np.uint8)
        colored_mask[mask] = [0, 255, 0]  # 마스크 색상 설정
        sample = cv2.addWeighted(sample, 1, colored_mask, 0.5, 0)

# 결과 이미지 출력
plt.imshow(sample)
plt.axis("off")
plt.show()
