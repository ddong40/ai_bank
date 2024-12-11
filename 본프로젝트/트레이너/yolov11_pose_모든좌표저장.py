from ultralytics import YOLO
import cv2
import os
import csv
import mediapipe as mp

# Load YOLO model
model = YOLO('yolov5/yolo11n.pt')

# MediaPipe 포즈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 로드
img_path = 'yolov5-master/yolov5-master/person.jpg'
frame = cv2.imread(img_path)

# YOLO 모델로 사람에 대한 바운딩 박스 생성
results = model(frame)

# 신뢰도가 가장 높은 바운딩 박스를 저장할 변수 초기화
best_box = None
best_confidence = 0.0

# CSV 파일을 저장할 경로 설정
csv_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/landmark_coordinates.csv'

# 랜드마크 ID와 좌표를 저장할 리스트 초기화
landmarks_data = []

if best_box is None:
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            confidence = box.conf[0]

            if int(class_id) == 0 and confidence > best_confidence:
                best_box = box
                best_confidence = confidence

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        person_roi = frame[y1:y2, x1:x2]

        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 각 랜드마크의 좌표를 landmarks_data에 교차 저장
                for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y
                    # 리스트에 x와 y 좌표를 교차로 저장
                    landmarks_data.append((f"x_{idx}", x))
                    landmarks_data.append((f"y_{idx}", y))

        frame[y1:y2, x1:x2] = person_roi

# CSV 파일에 저장
with open(csv_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # CSV 파일에 헤더 작성
    header = [f"x_{i}" for i in range(len(results_pose.pose_landmarks.landmark))] + [f"y_{i}" for i in range(len(results_pose.pose_landmarks.landmark))]
    csv_writer.writerow(header)

    # 랜드마크 데이터를 교차하여 저장
    row_data = []
    for idx in range(len(results_pose.pose_landmarks.landmark)):
        row_data.append(landmarks_data[idx * 2][1])  # x 값
        row_data.append(landmarks_data[idx * 2 + 1][1])  # y 값

    # CSV 파일에 한 줄로 저장
    csv_writer.writerow(row_data)

# 결과 프레임 화면에 표시
cv2.imshow("YOLOv11 - Person Detection with Pose", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
