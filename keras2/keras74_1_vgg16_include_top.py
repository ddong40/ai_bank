import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16

############################# 그냥 VGG16 모델 ##########################
# model = VGG16()
# model.summary()

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________


# VGG16 디폴트 파라미터 =VGG16()
# model = VGG16(weights='imagenet',           # 이미지넷 데이터 훈련 가중치
#               include_top=True,             # top = input,    True 면 top 까지 포함(위아래 포함)
#               input_shape=(224, 224, 3))    # 디폴트 파라미터

# model.summary()

model = VGG16(weights='imagenet',
              include_top=False,          
              input_shape=(100, 100, 3))    

model.summary()
################# include_top = False #######################
# 1. Fully Conneted layers(Dense layers) 를 날려버린다.
# 2. input_shape 를 내가 하고싶은 데이터 shape로 맞춰!!!!
