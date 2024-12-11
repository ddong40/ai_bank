## torch241 ##

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLOv11 모델 불러오기
model = YOLO('yolov5/yolo11n.pt')

# 이미지 불러오기
image_path = "C:/Users/ddong40/Desktop/paper/image/david.jpg"
image = cv2.imread(image_path)

# 객체 탐지 수행
results = model(image)

# 결과 시각화
annotated_image = results[0].plot()  # 결과가 반영된 이미지를 그립니다.

# 이미지 표시
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()