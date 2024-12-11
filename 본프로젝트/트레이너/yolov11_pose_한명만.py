from ultralytics import YOLO
import cv2
import os
import torch
# pip install mediapipe
import mediapipe as mp

# Load a model
model = YOLO('yolov5/yolo11n.pt') # 지정 경로에 있는 가중치 불러옴

# MediaPipe 포즈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 로드
img_path = 'yolov5-master/yolov5-master/person.jpg' # image 경로 설정
frame = cv2.imread(img_path)
        
# YOLO 모델로 사람에 대한 바운딩 박스 생성
results = model(frame)

# 신뢰도가 가장 높은 바운딩 박스를 저장할 변수 초기화
best_box = None
best_confidence = 0.0  # 최상의 신뢰도를 저장할 변수

if best_box is None:  # 아직 바운딩 박스가 선택되지 않은 경우
  
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]  # 클래스 ID 가져오기
            confidence = box.conf[0]  # 신뢰도 가져오기
            
            if int(class_id) == 0 and confidence > best_confidence:  # 사람 클래스 ID가 0이고, 현재 박스의 신뢰도가 최고일 경우
                best_box = box  # 현재 박스를 최상의 박스로 설정
                best_confidence = confidence  # 최상의 신뢰도 업데이트

    # 최상의 박스가 존재하면 해당 박스에 대해 작업 수행
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # 박스 좌표 저장
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 바운딩 박스 영역 추출
        person_roi = frame[y1:y2, x1:x2]
        
        # MediaPipe 포즈 추정
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))  # 포즈 랜드마크 처리
            
            # 포즈 랜드마크 그리기
            if results_pose.pose_landmarks:  # 랜드마크가 있는 경우 실행
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 랜드마크 그리기
                
        frame[y1:y2, x1:x2] = person_roi  # 포즈 결과를 원본 이미지에 반영
            
# 결과 프레임 화면에 표시
cv2.imshow("YOLOv11 - Person Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()