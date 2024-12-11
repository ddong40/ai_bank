import cv2
import mediapipe as mp
import math

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

## x,y 기반 각도 구하기 ##

def calculate_angle_from_landmarks(shoulder_landmark, pelvis_landmark):
    """
    Mediapipe 랜드마크를 사용하여 어깨, 골반, 어깨x-골반y 점을 기반으로 각도를 계산합니다.
    
    Parameters:
        shoulder_landmark: Mediapipe 랜드마크 데이터 (x, y 속성 포함)
        pelvis_landmark: Mediapipe 랜드마크 데이터 (x, y 속성 포함)
        
    Returns:
        angle_deg: 계산된 각도 (도 단위)
    """
    # 어깨 (A)와 골반 (B) 좌표 추출
    x1, y1 = shoulder_landmark.x, shoulder_landmark.y
    x2, y2 = pelvis_landmark.x, pelvis_landmark.y

    # 세 번째 좌표 C (어깨의 x, 골반의 y)
    x3, y3 = x1, y2

    # 벡터 AB와 BC 계산
    ab = (x2 - x1, y2 - y1)
    bc = (x3 - x2, y3 - y2)

    # 벡터 크기
    ab_magnitude = math.sqrt(ab[0]**2 + ab[1]**2)
    bc_magnitude = math.sqrt(bc[0]**2 + bc[1]**2)

    # 내적 계산
    dot_product = ab[0]*bc[0] + ab[1]*bc[1]

    # 각도 계산 (라디안 -> 도 단위 변환)
    angle_rad = math.acos(dot_product / (ab_magnitude * bc_magnitude))
    angle_deg = 180 - math.degrees(angle_rad) 

    return angle_deg



# 주어진 각도 계산 함수
def calculate_angle(point_a, point_b, point_c):
    ab = (point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z)
    bc = (point_c.x - point_b.x, point_c.y - point_b.y, point_c.z - point_b.z)
    dot_product = sum(a * b for a, b in zip(ab, bc))
    magnitude_ab = math.sqrt(sum(a**2 for a in ab))
    magnitude_bc = math.sqrt(sum(b**2 for b in bc)) 
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))

image_path = 'c:/Users/ddong40/Desktop/.PNG'

# 이미지 불러오기
image = cv2.imread(image_path)  # 'image_path'에 이미지 경로 입력

# 이미지를 RGB로 변환
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mediapipe를 사용하여 랜드마크 추출
result = pose.process(rgb_image)

# 랜드마크가 있을 경우 각도 계산
if result.pose_landmarks:
    landmarks = result.pose_landmarks.landmark
    # 어깨, 골반, 무릎 좌표 추출
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    # 각도 계산
    angle = calculate_angle_from_landmarks(shoulder, hip)
    print(f"Calculated angle: {angle:.2f} degrees")

# Mediapipe 종료
pose.close()

# 결과 이미지 출력
cv2.imshow('Pose Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

### xyz 좌표 기반 각도 함수 ##

# crunch up 각도 : 92.81 degrees

# crunch down 각도 : 80.53 degrees

## xy 기반 각도함수 ## 

# crunch down 각도 :  Calculated angle: 176.19 degrees

# crunch up 각도 : Calculated angle: 150.44 degrees

