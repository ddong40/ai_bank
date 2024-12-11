#수정필요

import cv2
import os
import csv
import mediapipe as mp
import math
from ultralytics import YOLO

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

# YOLO 모델 로드
model = YOLO('yolov5/yolo11n.pt')

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 비디오 파일 및 CSV 경로 설정
video_folder_path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/크런치_유튭/오자세/'
output_folder_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/output_folder/'
csv_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/z_kaggle_new_landmarks_incorrect_data.csv'

os.makedirs(output_folder_path, exist_ok=True)

# CSV 헤더 작성
header = [
    "Nose_x", "Nose_y", "Nose_z",
    "Left Eye_x", "Left Eye_y", "Left Eye_z",
    "Right Eye_x", "Right Eye_y", "Right Eye_z",
    "Left Ear_x", "Left Ear_y", "Left Ear_z",
    "Right Ear_x", "Right Ear_y", "Right Ear_z",
    "Left Shoulder_x", "Left Shoulder_y", "Left Shoulder_z",
    "Right Shoulder_x", "Right Shoulder_y", "Right Shoulder_z",
    "Left Elbow_x", "Left Elbow_y", "Left Elbow_z",
    "Right Elbow_x", "Right Elbow_y", "Right Elbow_z",
    "Left Wrist_x", "Left Wrist_y", "Left Wrist_z",
    "Right Wrist_x", "Right Wrist_y", "Right Wrist_z",
    "Left Hip_x", "Left Hip_y", "Left Hip_z",
    "Right Hip_x", "Right Hip_y", "Right Hip_z",
    "Left Knee_x", "Left Knee_y", "Left Knee_z",
    "Right Knee_x", "Right Knee_y", "Right Knee_z",
    "Left Ankle_x", "Left Ankle_y", "Left Ankle_z",
    "Right Ankle_x", "Right Ankle_y", "Right Ankle_z",
    "Neck_x", "Neck_y", "Neck_z",
    "Left Palm_x", "Left Palm_y", "Left Palm_z",
    "Right Palm_x", "Right Palm_y", "Right Palm_z",
    "Back_x", "Back_y", "Back_z",
    "Waist_x", "Waist_y", "Waist_z",
    "Left Foot_x", "Left Foot_y", "Left Foot_z",
    "Right Foot_x", "Right Foot_y", "Right Foot_z",
    "Left_Arm_Angle", "Right_Arm_Angle", "Body_Incline_Angle",
    "left_leg_angle", "right_leg_angle"
]

# CSV 파일 생성 및 헤더 작성
with open(csv_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)

    # 비디오 처리
    for video_file in os.listdir(video_folder_path):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_folder_path, video_file)
            output_video_path = os.path.join(output_folder_path, f"output_{video_file}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}.")
                continue

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # YOLO 결과
                    results = model(frame)
                    best_box = None
                    best_confidence = 0.0
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0 and box.conf[0] > best_confidence:
                                best_box = box
                                best_confidence = box.conf[0]

                    if best_box is not None:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                        person_roi = frame[y1:y2, x1:x2]
                        results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                        if results_pose.pose_landmarks:
                            landmarks = results_pose.pose_landmarks.landmark
                            row_data = []

                            # 랜드마크 좌표 추출
                            for i in [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]:
                                row_data.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])

                            # 추가 계산
                            neck_x = (landmarks[11].x + landmarks[12].x) / 2
                            neck_y = (landmarks[11].y + landmarks[12].y) / 2
                            neck_z = (landmarks[11].z + landmarks[12].z) / 2
                            row_data.extend([neck_x, neck_y, neck_z])

                            left_palm_x = (landmarks[19].x + landmarks[21].x + landmarks[17].x + landmarks[15].x) / 4
                            left_palm_y = (landmarks[19].y + landmarks[21].y + landmarks[17].y + landmarks[15].y) / 4
                            left_palm_z = (landmarks[19].z + landmarks[21].z + landmarks[17].z + landmarks[15].z) / 4
                            right_palm_x = (landmarks[20].x + landmarks[22].x + landmarks[18].x + landmarks[16].x) / 4
                            right_palm_y = (landmarks[20].y + landmarks[22].y + landmarks[18].y + landmarks[16].y) / 4
                            right_palm_z = (landmarks[20].z + landmarks[22].z + landmarks[18].z + landmarks[16].z) / 4
                            row_data.extend([left_palm_x, left_palm_y, left_palm_z, right_palm_x, right_palm_y, right_palm_z])

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
                            row_data.extend([back_x, back_y, back_z, waist_x, waist_y, waist_z])

                            # 거리 및 각도 계산 추가
                            # shoulder_distance = calculate_distance(landmarks[11], landmarks[12])
                            # left_arm_length = calculate_distance(landmarks[11], landmarks[13]) + calculate_distance(landmarks[13], landmarks[15])
                            # right_arm_length = calculate_distance(landmarks[12], landmarks[14]) + calculate_distance(landmarks[14], landmarks[16])
                            # left_leg_length = calculate_distance(landmarks[23], landmarks[25]) + calculate_distance(landmarks[25], landmarks[27])
                            # right_leg_length = calculate_distance(landmarks[24], landmarks[26]) + calculate_distance(landmarks[26], landmarks[28])
                            # leg_distance = calculate_distance(landmarks[27], landmarks[28])
                            left_arm_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                            right_arm_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                            body_incline_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                            left_leg_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                            right_leg_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])

                            row_data.extend([
                                left_arm_angle, right_arm_angle, body_incline_angle, left_leg_angle, right_leg_angle
                            ])

                            # 데이터 저장
                            csv_writer.writerow(row_data)

                            # 랜드마크 시각화
                            mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    out.write(frame)

            cap.release()
            out.release()

print("비디오 및 CSV 파일 생성 완료!")


