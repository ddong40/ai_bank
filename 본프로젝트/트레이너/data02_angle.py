## yolov11 ##
import cv2
import mediapipe as mp
import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np

#yolov11

# 미디어파이프 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 각도 계산 함수
def calculate_angle(a, b, c):
    """세 점의 각도를 계산하는 함수 (a, b, c는 각각 (x, y) 형태의 좌표)"""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 웹캠 입력 시작
video_path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/pushup_04.mp4'
cap = cv2.VideoCapture(video_path)

# font_path = 'NanumGothic.ttf'  # 사용할 한글 폰트 파일 경로
# font = ImageFont.truetype(font_path, 40)

feedback = ['Amazing!', 'You need to make your arm larger!', 'You need to make your arm smaller!']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 변환 및 포즈 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # 이미지 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 포즈 랜드마크 추출
    if results.pose_landmarks:
        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 좌표 가져오기
        landmarks = results.pose_landmarks.landmark
        
        # 어깨, 팔꿈치, 손목 좌표
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # print(shoulder)
        
        # 척추 중립 여부 판단하기 위한 어깨와 엉덩이의 기울기 비교
        shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
        hip_slope = abs(left_hip[1] - right_hip[1])
        
        if shoulder_slope < 0.05 and hip_slope < 0.05:  # 허용 오차는 필요에 따라 조정
            cv2.putText(image, 'back is good!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'back is not good!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        
        # if 40 < elbow_angle < 50:
        #     cv2.putText(image, feedback[0], (50, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # elif elbow_angle > 50:
        #     cv2.putText(image, feedback[2], (50, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # else :
        #     cv2.putText(image, feedback[1], (50, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)   


    # 화면에 이미지 출력
    cv2.imshow('Push-up Pose Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
