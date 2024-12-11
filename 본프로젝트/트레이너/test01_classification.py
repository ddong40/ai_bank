## yolov11 ##

import cv2
import os
import joblib
import mediapipe as mp
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import math
import xgboost as xgb

# 푸시업 레이블 정의
label = ['정자세', '오자세']

# YOLO 모델 불러오기
model = YOLO('yolov5/yolo11n.pt')

# MediaPipe 포즈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 푸시업 예측 모델 로드
crunch_model = joblib.load('c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/가중치/xgb_model.pkl')

# 비디오 파일 불러오기
video_path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/크런치_유튭/정자세/정자세4.mp4'
cap = cv2.VideoCapture(video_path)

# 한글 폰트 설정
font_path = "c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 40)

# 거리 계산 함수 (z 좌표 포함)
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

# 각도 계산 함수 (z 좌표 포함)
def calculate_angle(point_a, point_b, point_c):
    ab = (point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z)
    bc = (point_c.x - point_b.x, point_c.y - point_b.y, point_c.z - point_b.z)
    dot_product = sum(a * b for a, b in zip(ab, bc))
    magnitude_ab = math.sqrt(sum(a**2 for a in ab))
    magnitude_bc = math.sqrt(sum(b**2 for b in bc))
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))

# 비디오 반복 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 사람 바운딩 박스 생성
    results = model(frame)
    best_box = None
    best_confidence = 0.0

    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            confidence = box.conf[0]
            if int(class_id) == 0 and confidence > best_confidence:
                best_box = box
                best_confidence = confidence

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        person_roi = frame[y1:y2, x1:x2]

        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            landmarks_data = []

            if results_pose.pose_landmarks:
                try:
                    landmarks = results_pose.pose_landmarks.landmark

                    # 주요 랜드마크 추출 및 데이터 구성
                    for idx in [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]:
                        landmarks_data.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
                    # 추가 계산
                    neck_x = (landmarks[11].x + landmarks[12].x) / 2
                    neck_y = (landmarks[11].y + landmarks[12].y) / 2
                    neck_z = (landmarks[11].z + landmarks[12].z) / 2
                    landmarks_data.extend([neck_x, neck_y, neck_z])

                    left_palm_x = (landmarks[19].x + landmarks[21].x + landmarks[17].x + landmarks[15].x) / 4
                    left_palm_y = (landmarks[19].y + landmarks[21].y + landmarks[17].y + landmarks[15].y) / 4
                    left_palm_z = (landmarks[19].z + landmarks[21].z + landmarks[17].z + landmarks[15].z) / 4
                    right_palm_x = (landmarks[20].x + landmarks[22].x + landmarks[18].x + landmarks[16].x) / 4
                    right_palm_y = (landmarks[20].y + landmarks[22].y + landmarks[18].y + landmarks[16].y) / 4
                    right_palm_z = (landmarks[20].z + landmarks[22].z + landmarks[18].z + landmarks[16].z) / 4
                    landmarks_data.extend([left_palm_x, left_palm_y, left_palm_z, right_palm_x, right_palm_y, right_palm_z])

                    shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                    shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                    shoulder_mid_z = (landmarks[11].z + landmarks[12].z) / 2
                    hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                    hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                    hip_mid_z = (landmarks[23].z + landmarks[24].z) / 2

                    vector_x = hip_mid_x - shoulder_mid_x
                    vector_y = hip_mid_y - shoulder_mid_y
                    vector_z = hip_mid_z - shoulder_mid_z

                    back_x = shoulder_mid_x + (1/3) * vector_x
                    back_y = shoulder_mid_y + (1/3) * vector_y
                    back_z = shoulder_mid_z + (1/3) * vector_z
                    waist_x = shoulder_mid_x + (2/3) * vector_x
                    waist_y = shoulder_mid_y + (2/3) * vector_y
                    waist_z = shoulder_mid_z + (2/3) * vector_z
                    landmarks_data.extend([back_x, back_y, back_z, waist_x, waist_y, waist_z])


                    # 거리 및 각도 계산 추가
                    left_arm_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                    right_arm_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                    body_incline_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                    left_leg_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                    right_leg_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])

                    landmarks_data.extend([
                                left_arm_angle, right_arm_angle, body_incline_angle, left_leg_angle, right_leg_angle
                            ])

                    landmarks_dmatrix = xgb.DMatrix(np.array([landmarks_data], dtype=float))
                    # 푸시업 자세 예측
                    prediction = crunch_model.predict(landmarks_dmatrix)
                    prediction_class = (prediction > 0.5).astype(int)
                    prediction_text = label[int(prediction_class[0])]

                    # 결과 출력
                    print(f"Prediction Probabilities: {prediction}")
                    print(f"Prediction: {prediction_class}, Text: {prediction_text}")

                    # 바운딩 박스 및 텍스트 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((x1, y1 - 40), prediction_text, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                except IndexError:
                    print("IndexError: 랜드마크 데이터가 충분히 추출되지 않았습니다.")

    cv2.imshow("Push-Up Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
