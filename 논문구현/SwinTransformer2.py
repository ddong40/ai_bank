import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# UperNet + Swin Transformer 기반 Semantic Segmentation 모델 로드
segmentation_model_name = "openmmlab/upernet-swin-base"
image_processor = AutoImageProcessor.from_pretrained(segmentation_model_name)
segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(segmentation_model_name)

# Swin Transformer Classification 모델 로드
classification_model_name = "microsoft/swinv2-base-patch4-window8-256"
classification_feature_extractor = AutoFeatureExtractor.from_pretrained(classification_model_name)
classification_model = AutoModelForImageClassification.from_pretrained(classification_model_name)
classification_labels = classification_model.config.id2label

# 입력 이미지 준비
image_path = "C:/Users/ddong40/Desktop/paper/image/dog.jpg"
image = Image.open(image_path)

# 이미지가 PNG 형식이고 알파 채널이 있는 경우 RGB로 변환
if image.mode != "RGB":
    image = image.convert("RGB")

# Segmentation 수행
def segment_image(image):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = segmentation_model(**inputs)
        logits = outputs.logits  # [batch_size, num_classes, height, width]
        segmentation_map = logits.argmax(1).squeeze().cpu().numpy()
    return segmentation_map

# 사용자 선택 기반 Segmentation 필터링
def filter_segmentation_by_user_choice(segmentation_map, segmentation_labels):
    print("Available Segmentation Classes:")
    unique_classes = np.unique(segmentation_map)
    for class_id in unique_classes:
        if class_id in segmentation_labels:
            print(f"Class ID: {class_id}, Name: {segmentation_labels[class_id]}")

    while True:
        try:
            selected_class_id = int(input("Enter the Class ID you want to highlight: "))
            if selected_class_id in unique_classes:
                print(f"Selected Class: {segmentation_labels.get(selected_class_id,'Unknown')} (ID: {selected_class_id})")
                break
            else:
                print("Invalid Class ID. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid Class ID.")

    # 선택한 클래스만 필터링
    filtered_map = np.zeros_like(segmentation_map)
    filtered_map[segmentation_map == selected_class_id] = selected_class_id
    return filtered_map

# Segmentation 결과 병합 및 투명도 적용
def apply_segmentation_overlay(image, segmentation_map, alpha=0.5):
    # Segmentation 맵을 이미지 크기로 리사이즈
    target_size = image.size[::-1]  # PIL 이미지는 (width, height), Numpy는 (height, width)
    segmentation_map_resized = torch.nn.functional.interpolate(
        torch.tensor(segmentation_map).unsqueeze(0).unsqueeze(0).float(),
        size=target_size,
        mode="nearest",
    ).squeeze().numpy()

    # 컬러 맵 정의
    colormap = cm.get_cmap('jet', np.max(segmentation_map_resized) + 1)
    segmentation_overlay = colormap(segmentation_map_resized.astype(int))[..., :3]  # RGB만 사용

    # 원본 이미지를 NumPy 배열로 변환
    image_np = np.array(image) / 255.0  # 0~1 범위로 정규화

    # Segmentation Overlay 병합
    overlay = (1 - alpha) * image_np + alpha * segmentation_overlay
    overlay = np.clip(overlay, 0, 1)  # 값 클리핑
    return overlay

# Classification 수행
def classify_image(image):
    inputs = classification_feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = classification_labels[predicted_class_idx]
    return predicted_class_idx, predicted_class_name

# 시각화: 투명도가 적용된 Segmentation 결과
def visualize_results(image, segmentation_overlay, predicted_class_name):
    # 시각화
    plt.figure(figsize=(16, 8))

    # 원본 이미지와 Classification 결과
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Classification: {predicted_class_name}", fontsize=16)
    plt.axis("off")

    # Segmentation 결과
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_overlay)
    plt.title("Filtered Segmentation", fontsize=16)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 실행
predicted_class_idx, predicted_class_name = classify_image(image)
segmentation_map = segment_image(image)

# Segmentation 결과 필터링
segmentation_labels = segmentation_model.config.id2label
filtered_segmentation_map = filter_segmentation_by_user_choice(segmentation_map, segmentation_labels)

# Segmentation 결과 병합 및 시각화
segmentation_overlay = apply_segmentation_overlay(image, filtered_segmentation_map, alpha=0.5)
visualize_results(image, segmentation_overlay, predicted_class_name)
