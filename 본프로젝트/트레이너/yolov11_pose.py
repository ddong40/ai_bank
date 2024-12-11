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
img_path = 'yolov5-master/yolov5-master/bus.jpg' # image 경로 설정
frame = cv2.imread(img_path)
        
# YOLO 모델로 사람에 대한 바운딩 박스 생성
results = model(frame)

for result in results:
     for box in result.boxes:
         class_id = box.cls[0]  # 클래스 ID 가져오기
         
         if int(class_id) == 0:  # 사람 클래스 ID가 0이라고 가정
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #box.xyxy[0] 의 값을 각 변수에 정수형으로 저장
                    
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
            # (x1, y1)은 사각형 좌상단 모서리의 좌표 #(x2, y2)는 사각형 우하단 모서리 좌표 #(255, 0, 0)는 rgb값이다. 즉 파란색. #2는 사각형의 두께 즉 2픽셀
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 바운딩 박스 영역 추출
            person_roi = frame[y1:y2, x1:x2]
            
            # MediaPipe 포즈 추정
            with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose: # static_image_mode=True 정적 이미지에 대해서 포즈 추출, model_complexity=2 값이 높을 수록 정확도가 높아짐 0~2까지 값 가짐
                results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)) # 포즈 랜드마크 처리, 이 함수는 포즈에 대한 정보 포함하는 result_pose객체 반환
                
                # 포즈 랜드마크 그리기
                if results_pose.pose_landmarks: # 랜드마크가 있는 경우 실행
                    mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS) #draw_landmark는 result_pose에 랜드마크 그리고 랜드마크를 선으로 연결
            frame[y1:y2, x1:x2] = person_roi #포즈 결과를 원본 이미지에 반영
            
# 결과 프레임 화면에 표시
cv2.imshow("YOLOv11 - Person Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()