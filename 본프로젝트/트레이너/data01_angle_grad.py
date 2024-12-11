## yolov11##
import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
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

# 각도 계산 함수2
def calculate_angle_3d(a, b, c):
    """
    세 점의 각도를 계산하는 함수 (a, b, c는 각각 (x, y, z) 형태의 좌표)
    """
    # 벡터 AB와 BC 계산
    ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]

    # 벡터 AB와 BC의 내적과 크기 계산
    dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    # 각도 계산 (라디안 -> 도)
    angle = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    angle = abs(angle * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# 세 점이 일직선에 있는지 확인하는 함수
def is_straight_line(a, b, c, threshold=0.25):
    """a, b, c 세 점이 일직선에 가까운지 확인하는 함수.
    a, b, c는 각각 (x, y) 형태의 좌표.
    threshold는 기울기 차이 허용 오차.
    """
    # 어깨-골반과 골반-발목 기울기 계산
    slope_ab = (b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else float('inf')
    slope_bc = (c[1] - b[1]) / (c[0] - b[0]) if c[0] != b[0] else float('inf')
    
    # 기울기 차이가 threshold 이하인 경우 일직선으로 판단
    return abs(slope_ab - slope_bc) < threshold
'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/pushup_04.mp4'
'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/pushup_bad_01.mp4'
# 비디오 입력
video_path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/추가영상데이터/pushup_04.mp4'
cap = cv2.VideoCapture(video_path)

pushup_count = 0 # 푸시업 개수 변수
down_position = False # 아래로 내려간 상태 여부 변수

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 RGB로 변환하여 포즈 감지
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # 이미지를 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 포즈 랜드마크 추출
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 좌표 가져오기
        landmarks = results.pose_landmarks.landmark
        
        # 왼쪽 어깨, 골반, 발목 좌표 가져오기
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # 일직선 여부 확인
        if is_straight_line(left_shoulder, left_hip, left_ankle):
            pass
            # cv2.putText(image, 'Body is in straight line!', (50, 50), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Body is not in straight line.', (400, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        elbow_angle = calculate_angle(left_shoulder, elbow, wrist)    
        
        # 각도가 15도 이하로 내려갔을 때 상태 변경
        if elbow_angle < 85:
            down_position = True
        # 각도가 다시 올라왔을 때 푸시업 개수 증가
        elif down_position and elbow_angle > 90:
            pushup_count += 1
            down_position = False
            
        cv2.putText(image, f'Push-ups: {pushup_count}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        gyu_angle = calculate_angle_3d(left_hip, left_shoulder, elbow)
        
        # 각도 45도
        
        if 40 < gyu_angle < 45 :
            pass
        elif gyu_angle < 40 :
            cv2.putText(image, 'Spread your arms out more.', (600, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)    
        else :
            cv2.putText(image, 'Gather your arms more.', (600, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)      
        

    # 이미지 출력
    cv2.imshow('Straight Line Check', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
