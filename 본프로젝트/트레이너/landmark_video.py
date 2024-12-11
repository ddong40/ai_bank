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

# 비디오 파일이 있는 폴더 경로 설정
video_folder_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/'  # 비디오 파일이 있는 폴더 경로
output_folder_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/output_folder/'  # 결과 저장할 비디오 폴더 경로
os.makedirs(output_folder_path, exist_ok=True)

# CSV 파일 경로 설정
csv_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/landmarks_data.csv'

# CSV 파일을 열고 헤더 작성
with open(csv_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    header = ["Video", "Frame",
              "Nose_x", "Nose_y", "Left Eye_x", "Left Eye_y", "Right Eye_x", "Right Eye_y",
              "Left Ear_x", "Left Ear_y", "Right Ear_x", "Right Ear_y",
              "Left Shoulder_x", "Left Shoulder_y", "Right Shoulder_x", "Right Shoulder_y",
              "Left Elbow_x", "Left Elbow_y", "Right Elbow_x", "Right Elbow_y",
              "Left Wrist_x", "Left Wrist_y", "Right Wrist_x", "Right Wrist_y",
              "Left Hip_x", "Left Hip_y", "Right Hip_x", "Right Hip_y",
              "Left Knee_x", "Left Knee_y", "Right Knee_x", "Right Knee_y",
              "Left Ankle_x", "Left Ankle_y", "Right Ankle_x", "Right Ankle_y",
              "Neck_x", "Neck_y", "Left Palm_x", "Left Palm_y", "Right Palm_x", "Right Palm_y",
              "Back_x", "Back_y", "Waist_x", "Waist_y",
              "Left Foot_x", "Left Foot_y", "Right Foot_x", "Right Foot_y"]
    csv_writer.writerow(header)

    # 폴더 내 모든 비디오 파일에 대해 반복 처리
    for video_file in os.listdir(video_folder_path):
        if video_file.endswith('.mp4'):  # mp4 파일만 처리
            video_path = os.path.join(video_folder_path, video_file)
            output_video_path = os.path.join(output_folder_path, f"output_{video_file}")

            # 비디오 캡처 객체 생성
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}.")
                continue

            # 비디오 출력 설정 (해상도와 프레임 속도는 입력 비디오와 동일하게 설정)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # 비디오의 각 프레임을 반복 처리
            frame_idx = 0
            with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # 비디오가 끝나면 종료

                    # YOLO 모델로 사람에 대한 바운딩 박스 생성
                    results = model(frame)
                    
                    # 신뢰도가 가장 높은 바운딩 박스를 저장할 변수 초기화
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # 관심 영역(ROI) 추출
                        person_roi = frame[y1:y2, x1:x2]

                        # MediaPipe 포즈 인식
                        results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                        if results_pose.pose_landmarks:
                            mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                            # 랜드마크 좌표 추가
                            landmarks = results_pose.pose_landmarks.landmark
                            row_data = [video_file, frame_idx]

                            # 지정된 랜드마크 좌표 추가
                            row_data.extend([
                                landmarks[0].x, landmarks[0].y,  # Nose
                                landmarks[2].x, landmarks[2].y,  # Left Eye
                                landmarks[5].x, landmarks[5].y,  # Right Eye
                                landmarks[7].x, landmarks[7].y,  # Left Ear
                                landmarks[8].x, landmarks[8].y,  # Right Ear
                                landmarks[11].x, landmarks[11].y,  # Left Shoulder
                                landmarks[12].x, landmarks[12].y,  # Right Shoulder
                                landmarks[13].x, landmarks[13].y,  # Left Elbow
                                landmarks[14].x, landmarks[14].y,  # Right Elbow
                                landmarks[15].x, landmarks[15].y,  # Left Wrist
                                landmarks[16].x, landmarks[16].y,  # Right Wrist
                                landmarks[23].x, landmarks[23].y,  # Left Hip
                                landmarks[24].x, landmarks[24].y,  # Right Hip
                                landmarks[25].x, landmarks[25].y,  # Left Knee
                                landmarks[26].x, landmarks[26].y,  # Right Knee
                                landmarks[27].x, landmarks[27].y,  # Left Ankle
                                landmarks[28].x, landmarks[28].y,  # Right Ankle
                            ])

                            # 추가 좌표 계산
                            neck_x = (landmarks[11].x + landmarks[12].x) / 2
                            neck_y = landmarks[0].y - 0.05
                            row_data.extend([neck_x, neck_y])

                            left_palm_x = (landmarks[19].x + landmarks[21].x + landmarks[17].x + landmarks[15].x) / 4
                            left_palm_y = (landmarks[19].y + landmarks[21].y + landmarks[17].y + landmarks[15].y) / 4
                            right_palm_x = (landmarks[20].x + landmarks[22].x + landmarks[18].x + landmarks[16].x) / 4
                            right_palm_y = (landmarks[20].y + landmarks[22].y + landmarks[18].y + landmarks[16].y) / 4
                            row_data.extend([left_palm_x, left_palm_y, right_palm_x, right_palm_y])

                            shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                            shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                            hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                            hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                            vector_x = hip_mid_x - shoulder_mid_x
                            vector_y = hip_mid_y - shoulder_mid_y
                            back_x = shoulder_mid_x + (1/3) * vector_x
                            back_y = shoulder_mid_y + (1/3) * vector_y
                            waist_x = shoulder_mid_x + (2/3) * vector_x
                            waist_y = shoulder_mid_y + (2/3) * vector_y
                            row_data.extend([back_x, back_y, waist_x, waist_y])

                            # 발 좌표 추가
                            row_data.extend([
                                landmarks[31].x, landmarks[31].y,  # Left Foot
                                landmarks[32].x, landmarks[32].y   # Right Foot
                            ])

                            # 데이터 저장
                            csv_writer.writerow(row_data)

                    # 결과 비디오 프레임에 저장
                    out.write(frame)
                    frame_idx += 1

            # 비디오 종료
            cap.release()
           
