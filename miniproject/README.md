# 구글api 키 절대 유출 금지! (개인정보로써 1주일 뒤 삭제 예정)

# 연구프로젝트 : [AI 객체탐지 기술을 활용한 어선 선종 식별 기술 개발]

인공지능을 객체탐지 기능을 활용하여 서해의 대한민국 영해에 진입하는 어선에 대하여 국내, 중국 어선인지 분별하여 조업, 이동 여부를 확인합니다. 이는 해양 감시 업무에 효율성과 정확성을 높여 해양 안전과 경제적 이익을 도모합니다.

## 목차
1. [설치 방법](#설치-방법)
2. [사용 방법](#사용-방법)


## 설치방법

Python Version: Python 3.12.4
Torch Version : 2.4.1
CUDA Version: 12.4
CUDNN Version : 9.4
YOLOv11 Model File: yolov11n.pt

### 필요한 패키지 설치

pip install re
pip install opencv-pythton
pip install pandas 
pip install numpy
pip install geopy
pip install googlemaps
pip install matplotlib seaborn
pip install --upgrade google-cloud-vision
pip install ultralytics

## 사용방법 

### Custom 가중치 학습
train.py 

### 훈련된 가중치 모델 예측
model.predict.py 

### 구글 OCR API를 통한 이미지 내의 위치 좌표 도출
google_ocr.py

### Exif를 통해 제공된 이미지 Gps 정보 확인
Exif.py

### 최종 결과 테스트
AIDA_Test.py

### 최종 결과 도출
AIDA_Last_Result.py


