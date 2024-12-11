import cv2
import torch
from ultralytics import YOLO

# YOLOv10 모델 불러오기 (커스텀 가중치 사용)
model = YOLO('C:/Users/ddong40/ai_2/yolov10-main/yolov10-main/11x_best.pt')


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

    # 결과를 이미지 위에 표시 (plot() 메서드를 사용해 결과 시각화)
    result_img = results[0].plot()

    # 결과 이미지 출력
    cv2.imshow('YOLOv10 Object Detection', result_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap.release()
cv2.destroyAllWindows()