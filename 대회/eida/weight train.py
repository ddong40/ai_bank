from ultralytics import YOLO

# 모델 로드 (YOLOv11 프리트레인된 모델 사용, 또는 기본 YOLO 모델 설정)
def main():
    model = YOLO('C:/Users/ddong40/ai_2/_data/eida/ultralytics-main/ultralytics-main/yolo11x.pt')

# 모델 훈련

    model.train(data='C:/Users/ddong40/ai_2/_data/eida/seo 2.v1i.yolov11/data.yaml',  # 데이터셋 YAML 파일 경로
            epochs=50,                                # 훈련 에폭 수
            imgsz=640,                                # 이미지 크기
            batch=16,                                 # 배치 크기
            name='C:/Users/ddong40/ai_2/_data/eida/', # 훈련 결과 저장할 폴더 이름
            device='cuda' 
            )
if __name__ == '__main__':
    main()