import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False # vgg16 가중치 동결

model = Sequential()
model.add(vgg16) #얘는 훈련 안시켜
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax')) #얘네는 훈련 시켜

#1. 전체 동결

# model.trainable = False

#2. 전체동결
# for layer in model.layers: # 즉 model.layers는 이터레이터 형태로 구성되어 있다! 
#     layer.trainable = False


#3. 부분동결
# print(model.layers)
# [<keras.engine.functional.Functional object at 0x000002555363D910>, 
# <keras.layers.core.flatten.Flatten object at 0x00000255536369D0>, 
# <keras.layers.core.dense.Dense object at 0x00000255536B04F0>, 
# <keras.layers.core.dense.Dense object at 0x00000255536E1940>]

print(model.layers[2]) #Dense100 부분임
# <keras.layers.core.dense.Dense object at 0x00000210918405E0>

model.layers[2].trainable = False # flatten만 날림



# model.trainable = False
model.summary()

import pandas as pd
pd.set_option('max_colwidth', None) #pandas에서 제공되는 함수
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)


