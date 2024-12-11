from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load a model
model = YOLO('yolov5/yolo11n.pt')  # 지정 경로에 있는 가중치 불러옴

# MediaPipe 포즈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 동영상 로드
video_path = 'yolov5-master/yolov5-master/person.mp4'  # 동영상 경로 설정
cap = cv2.VideoCapture(video_path)

best_box = None
best_confidence = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 더 이상 프레임이 없으면 종료

    # YOLO 모델로 사람에 대한 바운딩 박스 생성
    results = model(frame)

    current_best_box = None  # 현재 프레임에서 가장 좋은 박스를 저장할 변수
    current_best_confidence = 0.0  # 현재 프레임에서 가장 높은 신뢰도 저장

    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]  # 클래스 ID 가져오기
            confidence = box.conf[0]  # 신뢰도 가져오기
            
            if int(class_id) == 0 and confidence > current_best_confidence:  # 사람 클래스 ID가 0이고, 현재 박스의 신뢰도가 최고일 경우
                current_best_box = box  # 현재 박스를 최상의 박스로 설정
                current_best_confidence = confidence  # 최상의 신뢰도 업데이트

    # 최상의 박스가 존재하면 해당 박스에 대해 작업 수행
    if current_best_box is not None:
        # best_box가 없거나 현재 프레임에서 찾은 박스가 더 높은 신뢰도를 가진 경우에만 업데이트
        if best_box is None or current_best_confidence > best_confidence:
            best_box = current_best_box  # 최상의 박스 업데이트
            best_confidence = current_best_confidence  # 최상의 신뢰도 업데이트

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # 박스 좌표 저장
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 바운딩 박스 영역 추출
        person_roi = frame[y1:y2, x1:x2]
        
        # MediaPipe 포즈 추정
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))  # 포즈 랜드마크 처리
            
            # 포즈 랜드마크 그리기
            if results_pose.pose_landmarks:  # 랜드마크가 있는 경우 실행
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 랜드마크 그리기
                
        frame[y1:y2, x1:x2] = person_roi  # 포즈 결과를 원본 이미지에 반영

    # 결과 프레임 화면에 표시
    cv2.imshow("YOLOv11 - Person Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 자원 해제
cap.release()
cv2.destroyAllWindows()

