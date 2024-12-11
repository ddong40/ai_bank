## tf274gpu ##

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Swin Transformer v2 Classification 모델
classification_model_name = "microsoft/swinv2-base-patch4-window8-256"
classification_feature_extractor = AutoFeatureExtractor.from_pretrained(classification_model_name)
classification_model = AutoModelForImageClassification.from_pretrained(classification_model_name)
classification_labels = classification_model.config.id2label

# YOLOv8 Segmentation 모델 로드
segmentation_model = YOLO("yolov8x-seg.pt")  # YOLOv8 모델 가중치 사용

# 입력 이미지 준비
image_path = "C:/Users/ddong40/Desktop/paper/image/dog.jpg"
image = Image.open(image_path)

# 이미지가 PNG 형식이고 알파 채널이 있는 경우 RGB로 변환
if image.mode != "RGB":
    image = image.convert("RGB")

# Classification 수행
def classify_image(image):
    inputs = classification_feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = classification_labels[predicted_class_idx]
    return predicted_class_idx, predicted_class_name

# YOLO Segmentation 수행
def segment_image_with_yolo(image):
    # OpenCV로 변환 (YOLO는 OpenCV BGR 이미지 필요)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = segmentation_model(image_bgr, task="segment")
    print("Detection results:", results)

    # 결과에서 마스크 및 클래스 이름 추출
    masks = results[0].masks.data.cpu().numpy()  # 마스크 배열
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 클래스 ID
    class_names = [results[0].names[cid] for cid in class_ids]  # 클래스 이름
    
    # YOLO로 검출된 클래스 출력
    print("Detected Classes:")
    for idx, class_name in enumerate(class_names):
        print(f"{idx}: {class_name}")

    # 사용자 입력을 통한 클래스 선택
    while True:
        try:
            selected_index = int(input("Enter the index of the class you want to highlight: "))
            if 0 <= selected_index < len(class_names):
                selected_class = class_names[selected_index]
                print(f"Selected Class: {selected_class}")
                break
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid index.")

    # 선택한 클래스 필터링
    segmentation_map = np.zeros_like(masks[0], dtype=np.uint8)
    for mask, class_name in zip(masks, class_names):
        if class_name == selected_class:  # 선택된 클래스만 병합
            segmentation_map = np.logical_or(segmentation_map, mask > 0.5).astype(np.uint8)

    return segmentation_map

# Segmentation 맵 시각화
def overlay_segmentation_with_yolo(image, segmentation_map, alpha=0.5):
    # OpenCV BGR 이미지로 변환
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Segmentation 맵 크기 조정 (YOLO 출력 크기 → 원본 이미지 크기)
    resized_map = cv2.resize(segmentation_map, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Segmentation 맵을 컬러로 변환
    colored_map = np.zeros_like(image_bgr, dtype=np.uint8)
    colored_map[resized_map > 0] = [255, 0, 0]  # 빨간색으로 표시

    # 이미지와 Segmentation 결과 병합
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_map, alpha, 0)
    return overlay

# 실행 및 시각화
predicted_class_idx, predicted_class_name = classify_image(image)
print(f"Predicted Class (Classification): {predicted_class_name}")

segmentation_map = segment_image_with_yolo(image)
segmentation_overlay = overlay_segmentation_with_yolo(image, segmentation_map)

# Classification 및 Segmentation 결과 시각화
plt.figure(figsize=(16, 8))

# 원본 이미지와 Classification 결과
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title(f"Classification: {predicted_class_name}", fontsize=16)
plt.axis("off")

# Segmentation 결과
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmentation_overlay, cv2.COLOR_BGR2RGB))  # BGR → RGB 변환
plt.title("Segmentation Overlay", fontsize=16)
plt.axis("off")

plt.tight_layout()
plt.show()
