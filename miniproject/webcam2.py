import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLOv10 모델 불러오기 (커스텀 가중치 사용)
model = YOLO('C:/Users/ddong40/ai_2/yolov10-main/yolov10-main/11x_best.pt')

# 색상 매핑 (예시로 5개의 클래스 색상)
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')  # 80개의 클래스에 대한 무작위 색상 생성

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 비디오 스트림 읽기 및 객체 탐지
while True:
    ret, frame = cap.read()
    if not ret:
        print("캠을 열 수 없습니다!")
        break

    # YOLO 모델을 통해 객체 탐지 수행
    results = model.predict(frame)

    # 탐지된 결과에서 boxes 정보 얻기
    boxes = results[0].boxes  # 탐지된 객체들의 bounding boxes 정보
    names = results[0].names  # 클래스 이름 정보

    # confidence score가 0.7 이상인 결과만 필터링
    filtered_boxes = boxes[boxes.conf >= 0.7]

    # 필터링된 결과를 이미지 위에 수동으로 표시
    result_img = frame.copy()
    for i in range(len(filtered_boxes.xyxy)):  # 각 bounding box에 대해
        box = filtered_boxes.xyxy[i]  # i번째 bounding box
        conf = round(filtered_boxes.conf[i].item(), 2)  # confidence score
        class_id = int(filtered_boxes.cls[i].item())  # 클래스 ID
        class_name = names[class_id]  # 클래스 이름 가져오기
        x1, y1, x2, y2 = map(int, box)  # 좌표를 정수로 변환
        
        # 클래스 ID에 해당하는 색상 선택
        color = colors[class_id]

        # 바운딩 박스와 클래스 이름 표시
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color.tolist(), 2)  # 박스 그리기
        cv2.putText(result_img, f'{class_name} {conf}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 클래스 이름 및 confidence score 표시

    # 결과 이미지 출력
    cv2.imshow('YOLOv10 Object Detection', result_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap.release()
cv2.destroyAllWindows()