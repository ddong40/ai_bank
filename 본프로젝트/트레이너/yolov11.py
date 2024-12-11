from ultralytics import YOLO
import cv2
import os
import torch

# Load a model
model = YOLO('yolov5/yolo11n.pt')

img_path = 'yolov5-master/yolov5-master/bus.jpg'



# 이미지 로드
frame = cv2.imread(img_path)
    
# YOLO 모델로 사람에 대한 바운딩 박스 생성
results = model(frame)

for result in results:
     for box in result.boxes:
         class_id = box.cls[0]  # 클래스 ID 가져오기
         if int(class_id) == 0:  # 사람 클래스 ID가 0이라고 가정
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            

# 결과 프레임 화면에 표시
cv2.imshow("YOLOv11 - Person Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()