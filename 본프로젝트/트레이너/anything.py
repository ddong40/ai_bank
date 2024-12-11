import tensorflow as tf

# TensorFlow 버전 출력
print("TensorFlow Version:", tf.__version__)

# GPU가 사용 가능한지 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU가 {len(gpus)}개 감지되었습니다:")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("GPU를 찾을 수 없습니다.")