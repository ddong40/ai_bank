from ultralytics import YOLO
import cv2
import torch
import pandas as pd

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# YOLO 모델 로드
model = YOLO("yolov10x.pt")
# model.to('cuda')