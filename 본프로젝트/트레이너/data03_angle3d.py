import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 각도 계산 함수 (3D 좌표)
def calculate_angle_3d(a, b, c):
    """세 점의 각도를 계산하는 함수 (a, b, c는 각각 (x, y, z) 형태의 좌표)"""
    ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]

    dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    angle = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    angle = abs(angle * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/pushup_02.mp4'

# 비디오 파일 입력
video_path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/pushup_02.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 변환 및 포즈 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # 이미지 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 포즈 랜드마크 추출 및 각도 계산
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        
        # Mediapipe의 랜드마크 좌표 가져오기
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        
        # 각도 계산
        elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_hip)
        
        # 각도 출력
        cv2.putText(image, f'Elbow Angle: {int(elbow_angle)} degrees', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 이미지 출력
    cv2.imshow('Push-up Angle Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
