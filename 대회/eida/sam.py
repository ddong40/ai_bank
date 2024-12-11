import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# SAM 모델 로드
sam_checkpoint = 'segment-anything-main/segment-anything-main/sam_vit_h_4b8939.pth'  # SAM 모델 체크포인트 파일 경로
model_type = "vit_h"  # 사용할 모델 타입 (예: vit_h, vit_l, vit_b)
device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM 모델을 로드합니다.
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 세그멘테이션을 수행할 이미지 로드
input_image_path = 'segment-anything-main/segment-anything-main/notebooks/images/dog.jpg'  # 어종 이미지 경로
image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 포맷을 사용하므로 RGB로 변환

# 이미지 입력 및 세그멘테이션
predictor.set_image(image)

# 자동으로 여러 개의 마스크 생성
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=None,
    multimask_output=True  # 여러 개의 마스크 생성
)

# 마스크에 라벨을 부여한 이미지 생성
labeled_image = image.copy()
for i, mask in enumerate(masks):
    # 마스크 영역에 색상 적용 (각 마스크에 대해 다른 색상)
    color = (0, 255 // (i + 1), 255 - (255 // (i + 1)))  # 예시 색상
    labeled_image[mask] = labeled_image[mask] * 0.5 + np.array(color) * 0.5

    # 마스크의 중앙에 라벨 텍스트 추가
    y, x = np.where(mask)
    if len(x) > 0 and len(y) > 0:
        center = (int(np.mean(x)), int(np.mean(y)))
        cv2.putText(labeled_image, f"Object {i+1}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# 결과를 저장
output_image_path = "C:/Users/ddong40/ai_2/images/labeled_fish.png"
cv2.imwrite(output_image_path, cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))

print(f"라벨이 부여된 세그멘테이션 결과가 {output_image_path}에 저장되었습니다.")
