from ultralytics import YOLO
import cv2
import torch
# import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLO 모델 로드
model = YOLO("yolo11n.pt") 

# 이미지에서 예측 수행

def main():
    model.to('cuda').train(data = '_data/eida/seo 2.v1i.yolov11/data.yaml',
                                    epochs = 50, patience= 30 ,batch= 2, device = 0, imgsz = 1024)

if __name__ == "__main__":
    main()



# # 결과를 플롯으로 시각화
# plots = result[0].plot()
# cv2.imshow("plot", plots)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
