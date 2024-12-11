import torch
import cv2
import numpy as np


#1. yolo 10으로 훈련된 custom 가중치를 불러오기

custom_weights_path = 'path'
model = torch.hub.load('ultralytics/yollov10', 'custom', path=custom_weights_path, source='local' )

# 라벨 매핑 설정
label_mapping = {
    'china_bumjangmang_fishing': '범장망 조업',
    'china_bumjangmang_moving': '범장망 이동',
    'china_bumjangmang_poryu': '범장망 포류',
    'china_deunggwangjo_moving': '등광조망 조업',
    'china_deunggwnagjo_poryu': '등광조망 이동',
    'china_deungwangjo_fishing': '등광조망 포류',
    'china_tamang_fishing' : '타망 조업',
    'china_tamang_moving' : '타망 이동',
    'china_tamang_poryu' : '타망 포류',
    'china_youmang_fishing' : '유망 조업',
    'china_youmang_moving' : '유망 이동',
    'china_youmang_poryu' : '유망 포류',
    'korea_angangmang_fishing' : '안강망 조업',
    'korea_angangmang_moving' : '안강망 이동',
    'korea_angangmang_poryu' : '안강망 포류',
    'korea_chaenakgi_fishing' : '채낚기 조업',
    'korea_chaenakgi_moving' : '채낚기 이동',
    'korea_chaenakgi_poryu' : '채낚기 포류',
    'korea_jeoinmang_fishing' : '저인망 조업',
    'korea_jeoinmang_moving' : '저인망 이동',
    'korea_jeoinmang_poryu' : '저인망 포류',
    'korea_nakshieoseon_fishing' : '낚시선 조업',
    'korea_nakshieoseon_moving' : '낚시선 이동',
    'korea_nakshieoseon_poryu' : '낚시선 포류',
    'korea_tongbal_fishing' : '통발 조업',
    'korea_tongbal_moving' : '통발 이동',
    'korea_tongbal_poryu' : '통발 포류',
    'korea_yeonseung_fishing' : '연승 조업',
    'korea_yeonseung_moving' : '연승 이동',
    'korea_yeonseung_poryu' : '연승 포류'
    }

# "국내" 또는 "중국"을 구분할 라벨 인덱스 설정 (예: 0~17번까지는 "국내", 나머지는 "중국")
domestic_labels = list(range(12))  # "중국"로 표시할 라벨 인덱스    

# 이미지 경로
image_path = 'path'

# 이미지 로드
image = cv2.imread(image_path)

# 모델을 사용하여 객체 감지 수행
results = model(image)

# 결과에서 탐지된 객체 정보 가져오기
detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class

# 감지된 객체 표시
for detection in detections:
    x1, y1, x2, y2, confidence, cls = map(int, detection[:6])
    label = results.names[cls]  # 탐지된 라벨 이름
    
    # 라벨 매핑을 통해 한국어 라벨 가져오기
    korean_label = label_mapping.get(label, label)  # 매핑되지 않은 라벨은 기본값 사용
    
    # 조건에 따른 라벨링
    if cls in domestic_labels:
        label_text = f"중국 {korean_label}"
    else:
        label_text = f"대한민국 {korean_label}"

    # 바운딩 박스 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 라벨 텍스트 출력
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# 결과 이미지 출력
cv2.imshow('YOLOv10 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




#3. 
# 모델 구성, 라벨이 0-11이면 중국, 12-29면 대한민국을 상자 위에 호출
# 선종 호출
# 함께 조업여부 출력
# 좌표값 출력여부? 

