import torch
import threading
import cv2
import numpy as np
import pathlib
from ultralytics import YOLO  # YOLOv8 라이브러리 추가
import requests
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 모델 로드
model_v8 = YOLO('yolov8l.pt')  # YOLOv8 모델 불러오기

min_confidence = 0.5  # 최소 confidence 설정
iou_threshold = 0.5  # IoU 기준 중복 판단 임계값

# 영상 경로 (CCTV 스트림 또는 파일 경로)
video_path = "http://192.168.0.109:5000/video_feed"  # CCTV 스트림

# 비디오 캡처
cap = cv2.VideoCapture('http://192.168.0.109:5000/video_feed')
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# 이벤트 객체 생성 (값이 변경되면 신호를 줌)
value_changed_event = threading.Event()

# 전역 변수로 이전 값과 현재 값을 저장
previous_car_count = None
previous_parking_lot = None

current_car_count = 0
current_parking_lot = 20  # 주차 공간 20개

# YOLOv8 결과 처리 및 주차 공간 계산
def process_yolov8_results(results_v8, frame):
    global current_car_count, current_parking_lot
    car_count = 0
    parking_lot = 8  # 주차 공간

    # YOLOv8에서 결과 가져오기
    v8_boxes = results_v8[0].boxes.xyxy.cpu()  # 바운딩 박스 좌표
    v8_conf = results_v8[0].boxes.conf.cpu()  # confidence score
    v8_classes = results_v8[0].boxes.cls.cpu()  # 클래스 ID

    final_boxes, final_conf, final_classes = [], [], []

    # YOLOv8의 car, truck, bus만 처리
    for i, v8_box in enumerate(v8_boxes):
        if v8_classes[i] in [0, 1, 2]:  # YOLOv8에서 'car', 'truck', 'bus'
            final_boxes.append(v8_box)
            final_conf.append(v8_conf[i])
            final_classes.append(v8_classes[i])

    # 탐지된 객체를 프레임에 그리기 및 차량 수 세기
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(final_classes[i])

        if class_id == 0:  # car
            label = f"car: {final_conf[i]:.2f}"
            color = (0, 255, 0)  # green for car
            car_count += 1
        elif class_id == 1:  # truck
            label = f"truck: {final_conf[i]:.2f}"
            color = (255, 0, 0)  # blue for truck
            car_count += 1
        elif class_id == 2:  # bus
            label = f"bus: {final_conf[i]:.2f}"
            color = (0, 0, 255)  # red for bus
            car_count += 1
        else:
            label = f"Unknown class: {final_conf[i]:.2f}"
            color = (0, 255, 255)  # yellow for unknown class

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 주차장 빈 공간 업데이트
    current_car_count = car_count
    current_parking_lot = parking_lot - car_count

    return frame

# YOLOv8 탐지 및 결과 처리
def detectAndDisplay(frame):
    # YOLOv8 결과
    results_v8 = model_v8(frame)

    # 결과 처리 및 프레임에 그리기
    frame = process_yolov8_results(results_v8, frame)

    # 결과 이미지 출력
    cv2.imshow("YOLOv8 Detection", frame)

# 값이 변경되었을 때 처리하는 함수
def handle_value_change():
    global previous_car_count, previous_parking_lot, current_car_count, current_parking_lot
    while True:
        # 이벤트가 발생할 때까지 대기
        value_changed_event.wait()

        # Flask 서버로 데이터를 전송
        data = {
            "car_count": current_car_count,
            "empty_spots": current_parking_lot
        }
        response = requests.post("http://localhost:5000/update_parking_info", json=data)
        print(f"서버 응답: {response.text}")

        # 이전 값을 현재 값으로 갱신
        previous_car_count = current_car_count
        previous_parking_lot = current_parking_lot

        # 이벤트를 다시 초기화
        value_changed_event.clear()

# 영상 처리 루프
def video_loop():
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame)

        # 값이 변경되었을 때 이벤트 발생
        if current_car_count != previous_car_count or current_parking_lot != previous_parking_lot:
            value_changed_event.set()  # 이벤트 신호를 보냄

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 두 스레드를 실행: 하나는 영상을 처리하고, 하나는 Flask 서버로 데이터를 전송하는 역할
monitor_thread = threading.Thread(target=video_loop)
handler_thread = threading.Thread(target=handle_value_change)

monitor_thread.start()
handler_thread.start()

monitor_thread.join()
handler_thread.join()

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
