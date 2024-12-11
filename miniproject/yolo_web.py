import cv2
import torch
from ultralytics import YOLO
path='C:/Users/ddong40/ai_2/yolov5/best.pt'
# YOLO 모델 로드 (커스텀 가중치)
model =YOLO('C:/Users/ddong40/ai_2/yolov5/best.pt')

# 웹캠 설정
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    # 이미지에서 객체 탐지
    results = model(frame)

    # 탐지 결과 가져오기
    detections = results[0]  # 첫 번째 프레임의 결과

    for detection in detections.boxes:
        # 바운딩 박스 정보 가져오기
        xyxy = detection.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표를 NumPy 배열로 변환
        x1, y1, x2, y2 = map(int, xyxy)  # 좌표를 정수로 변환
        conf = detection.conf.item()  # 신뢰도 (item() 메서드로 스칼라 값으로 변환)
        cls = int(detection.cls.item())  # 클래스 인덱스 (item() 메서드로 스칼라 값으로 변환)

        # 바운딩 박스와 클래스 레이블 그리기
        label = f'Object {cls}: {conf:.2f}'  # 클래스 이름이 없으므로 숫자로 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 프레임 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()