import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor  # SAM2 라이브러리

# 모델 로드
sam2_model_id = "facebook/sam2-hiera-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = SAM2ImagePredictor.from_pretrained(sam2_model_id)

# 이미지 경로 설정
image_path = r"C:/Users/noori/Pictures/사진/KakaoTalk_20241105_165634523.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)


# 이미지 로드 및 전처리
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

input_points = np.array([[480, 500], [490, 910]])  # 사용자가 선택한 점
input_labels = np.array([1, 1])  # 1: 포그라운드, 0: 백그라운드


# 세그멘테이션 수행
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    sam2.set_image(image_np)    
    masks, _, _ = sam2.predict(point_coords=input_points, point_labels=input_labels)

# 마스크 병합
if len(masks) > 0:
    all_masks = np.any(masks > 0, axis=0).astype(np.uint8)  # 마스크 병합
else:
    raise ValueError("No masks were generated.")

# 시각화
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_np)
plt.imshow(all_masks, alpha=0.5, cmap="cool")  # 색상 맵 개선
plt.title("Segmented Image with SAM2")
plt.axis("off")

plt.show()
