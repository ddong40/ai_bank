from ultralytics import YOLO
import cv2
import torch
import pandas as pd

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

custom_weights_path = "C:/Users/ddong40/ai_2/factory/weights/best.pt"

# YOLO 모델 로드
model = YOLO(custom_weights_path) 

# 이미지에서 예측 수행
result = model.predict('C:/Users/ddong40/ai_2/factory/data/209/*.png', save=True, conf=0.7)

# 결과를 플롯으로 시각화
plots = result[0].plot()


# 탐지된 객체 좌표를 CSV 파일로 저장
boxes = result[0].boxes.xyxy  # 탐지된 객체의 좌표 (xmin, ymin, xmax, ymax)
scores = result[0].boxes.conf  # 신뢰도 점수
classes = result[0].boxes.cls  # 탐지된 클래스
# 데이터 프레임 생성
data = {
    'xmin': boxes[:, 0].cpu().numpy(),
    'ymin': boxes[:, 1].cpu().numpy(),
    'xmax': boxes[:, 2].cpu().numpy(),
    'ymax': boxes[:, 3].cpu().numpy(),
    'confidence': scores.cpu().numpy(),
    'class': classes.cpu().numpy()
}

df = pd.DataFrame(data)

print(df)


# CSV 파일로 저장
df.to_csv("C:/Users/ddong40/ai_2/factory/submit/418_detected.csv", index=False)

cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()