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
        cv2.rectangle(frame, (x1, y1), (x2, y1), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        person_roi = frame[y1:y2, x1:x2]

        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Right Knee (26)와 Left Knee (25) 좌표를 landmarks_data에 저장
                nose = results_pose.pose_landmarks.landmark[0]
                left_eye = results_pose.pose_landmarks.landmark[2]
                right_eye = results_pose.pose_landmarks.landmark[5]
                left_ear = results_pose.pose_landmarks.landmark[7]
                right_ear = results_pose.pose_landmarks.landmark[8]
                left_shoulder = results_pose.pose_landmarks.landmark[11]
                right_shoulder = results_pose.pose_landmarks.landmark[12]
                left_elbow = results_pose.pose_landmarks.landmark[13]
                right_elbow = results_pose.pose_landmarks.landmark[14]
                left_wrist = results_pose.pose_landmarks.landmark[15]
                right_wrist = results_pose.pose_landmarks.landmark[16]
                left_pinky = results_pose.pose_landmarks.landmark[17]
                right_pinky = results_pose.pose_landmarks.landmark[18]
                left_index = results_pose.pose_landmarks.landmark[19]
                right_index = results_pose.pose_landmarks.landmark[20]
                left_thumb = results_pose.pose_landmarks.landmark[21]
                right_thumb = results_pose.pose_landmarks.landmark[22]
                left_hip = results_pose.pose_landmarks.landmark[23]
                right_hip = results_pose.pose_landmarks.landmark[24]
                left_knee = results_pose.pose_landmarks.landmark[25] # Left Knee
                right_knee = results_pose.pose_landmarks.landmark[26]  # Right Knee
                  
                left_ankle = results_pose.pose_landmarks.landmark[27]
                right_ankle = results_pose.pose_landmarks.landmark[28]
                
                left_heel = results_pose.pose_landmarks.landmark[29]
                right_heel = results_pose.pose_landmarks.landmark[30]
                left_foot = results_pose.pose_landmarks.landmark[31]
                right_foot  = results_pose.pose_landmarks.landmark[32]
                
            
            
            
                landmarks_data.append((f"Nose_x", nose.x)) # [0][1]
                landmarks_data.append((f"Nose_y", nose.y)) # [1][1]
                landmarks_data.append((f"Left Eye_x", left_eye.x)) # [2][1]
                landmarks_data.append((f"Left Eye_y", left_eye.y)) # [3][1]
                landmarks_data.append((f"Right Eye_x", right_eye.x)) # [4][1]
                landmarks_data.append((f"Right Eye_y", right_eye.y)) # [5][1]
                landmarks_data.append((f"Left Ear_x", left_ear.x)) # [6][1]
                landmarks_data.append((f"Left Ear_y", left_ear.y)) # [7][1]
                landmarks_data.append((f"Right Ear_x", right_ear.x)) # [8][1]
                landmarks_data.append((f"Right Ear_y", right_ear.y)) # [9][1]
                landmarks_data.append((f"Left Shoulder_x", left_shoulder.x)) # [10][1]
                landmarks_data.append((f"Left Shoulder_y", left_shoulder.y)) # [11][1]
                landmarks_data.append((f"Right Shoulder_x", right_shoulder.x)) # [12][1]
                landmarks_data.append((f"Right Shoulder_y", right_shoulder.y)) # [13][1]
                
                landmarks_data.append((f"Left Elbow_x", left_elbow.x)) # [14][1]
                landmarks_data.append((f"Left Elbow_y", left_elbow.y)) # [15][1]
                
                landmarks_data.append((f"Right Elbow_x", right_elbow.x)) # [16][1]
                landmarks_data.append((f"Right Elbow_y", right_elbow.y)) # [17][1]
                
                landmarks_data.append((f"Left Wrist_x", left_wrist.x)) # [18][1]
                landmarks_data.append((f"Left Wrist_y", left_wrist.y)) # [19][1]
                landmarks_data.append((f"Right Wrist_x", right_wrist.x)) # [20][1]
                landmarks_data.append((f"Right Wrist_y", right_wrist.y)) # [21][1]
                landmarks_data.append((f"Left Hip_x", left_hip.x)) # [22][1]
                landmarks_data.append((f"Left Hip_y", left_hip.y)) # [23][1]
                
                landmarks_data.append((f"Right Hip_x", right_hip.x)) # [24][1]
                landmarks_data.append((f"Right Hip_y", right_hip.y)) # [25][1]
                
                landmarks_data.append((f"Left Knee_x", left_knee.x)) # [26][1]
                landmarks_data.append((f"Left Knee_y", left_knee.y)) # [27][1]
                landmarks_data.append((f"Right Knee_x", right_knee.x)) # [28][1]
                landmarks_data.append((f"Right Knee_y", right_knee.y)) # [29][1]
                landmarks_data.append((f"Left Ankle_x", left_ankle.x)) # [30][1]
                landmarks_data.append((f"Left Ankle_y", left_ankle.y)) # [31][1]
                
                landmarks_data.append((f"Right Ankle_x", right_ankle.x)) # [32][1]
                landmarks_data.append((f"Right Ankle_y", right_ankle.y)) # [33][1]
                
                landmarks_data.append((f"Neck_x", (left_shoulder.x + right_shoulder.x)/2)) # [34][1]
                landmarks_data.append((f"Neck_y", nose.y - 0.05)) # [35][1]
                
                landmarks_data.append((f"Left Palm_x", (left_index.x + left_thumb.x + left_pinky.x + left_wrist.x)/4)) #각 좌표의 무게중심, 즉 손바닥 랜드마크 생성 # [36][1]
                landmarks_data.append((f"Left Palm_y", (left_index.y + left_thumb.y + left_pinky.y + left_wrist.y)/4))  # [37][1]
                landmarks_data.append((f"Right Palm_x", (right_index.x + right_thumb.x + right_pinky.x + right_wrist.x)/4)) #각 좌표의 무게중심, 즉 손바닥 랜드마크 생성 # [38][1]
                landmarks_data.append((f"Right Palm_y", (right_index.y + right_thumb.y + right_pinky.y + right_wrist.y)/4)) # [39][1]
                
                # 어깨와 엉덩이의 중앙점 계산
                shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 
                shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2

                hip_mid_x = (left_hip.x + right_hip.x) / 2
                hip_mid_y = (left_hip.y + right_hip.y) / 2

                # 어깨에서 엉덩이까지의 벡터
                vector_x = hip_mid_x - shoulder_mid_x
                vector_y = hip_mid_y - shoulder_mid_y

                
                landmarks_data.append((f"Back_x", shoulder_mid_x +(1/3)*vector_x)) # 어깨와 엉덩이뼈 1/3내분점 # [40][1]
                landmarks_data.append((f"Back_y", shoulder_mid_y +(1/3)*vector_y)) # [41][1]
                landmarks_data.append((f"Waist_x", shoulder_mid_x +(2/3)*vector_x)) # 어깨와 엉덩이뼈 2/3내분점 # [42][0]
                landmarks_data.append((f"Waist_y", shoulder_mid_y +(2/3)*vector_y)) # [43][1]
                
                landmarks_data.append((f"Left Foot_x", left_foot.x)) # [44][0]
                landmarks_data.append((f"Left Foot_y", left_foot.y)) # [45][1]
                landmarks_data.append((f"Right Foot_x", right_foot.x)) # [46][0]
                landmarks_data.append((f"Right Foot_y", right_foot.y)) # [47][1]
                
        frame[y1:y2, x1:x2] = person_roi

# CSV 파일에 저장
with open(csv_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # CSV 파일에 헤더 작성
    header = ["Nose_x", "Nose_y", "Left Eye_x", "Left Eye_y", 'Right Eye_x', 'Right Eye_y', 'Left Ear_x', 'Left Ear_y', 'Right Ear_x', 
              'Right Ear_y', 'Left Shoulder_x', 'Left Shoulder_y', 'Right Shoulder_x', 'Right Shoulder_y', 'Left Elbow_x', 'Left Elbow_y', 'Right Elbow_x', 'Right Elbow_y',
              'Left Wrist_x', 'Left Wrist_y', 'Right Wrist_x', 'Right Wrist_y', 'Left Hip_x', 'Left Hip_y', 'Right Hip_x', 'Right Hip_y', 'Left Knee_x', 'Left Knee_y',
              'Right Knee_x', 'Right Knee_y', 'Left Ankle_x', 'Left Ankle_y', 'Right Ankle_x', 'Right Ankle_y', 'Neck_x', 'Neck_y', 'Left Palm_x', 'Left Palm_y',
              'Right Palm_x', 'Right Palm_y', 'Back_x', 'Back_y', 'Waist_x', 'Waist_y', 'Left Foot_x', 'Left Foot_y', 'Right Foot_x', 'Right Foot_y']
    csv_writer.writerow(header)

    # 랜드마크 데이터를 저장
    row_data = [landmarks_data[0][1], landmarks_data[1][1], landmarks_data[2][1], landmarks_data[3][1], landmarks_data[4][1], landmarks_data[5][1], landmarks_data[6][1],
                landmarks_data[7][1], landmarks_data[8][1], landmarks_data[9][1], landmarks_data[10][1], landmarks_data[11][1], landmarks_data[12][1], landmarks_data[13][1],
                landmarks_data[14][1], landmarks_data[15][1], landmarks_data[16][1], landmarks_data[17][1], landmarks_data[18][1], landmarks_data[19][1], landmarks_data[20][1], 
                landmarks_data[21][1], landmarks_data[22][1], landmarks_data[23][1], landmarks_data[24][1], landmarks_data[25][1], landmarks_data[26][1], landmarks_data[27][1], 
                landmarks_data[28][1], landmarks_data[29][1], landmarks_data[30][1], landmarks_data[31][1], landmarks_data[32][1], landmarks_data[33][1], landmarks_data[34][1], 
                landmarks_data[35][1], landmarks_data[36][1], landmarks_data[37][1], landmarks_data[38][1], landmarks_data[39][1], landmarks_data[40][1], landmarks_data[41][1],
                landmarks_data[42][1], landmarks_data[43][1], landmarks_data[44][1], landmarks_data[45][1], landmarks_data[46][1], landmarks_data[47][1]]  # x, y 좌표 추출
    csv_writer.writerow(row_data)

# 결과 프레임 화면에 표시
cv2.imshow("YOLOv11 - Person Detection with Pose", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
