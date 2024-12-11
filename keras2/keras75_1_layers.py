import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.trainable = False
model.summary()

print(model.weights)
exit()
print(len(model.weights)) #layer당 길이가 2개가 나옴, 가중치와 bias 하나씩 계산했을 때 
print(len(model.trainable_weights))

#30인 이유는  include top = False로 둠에 따라 fully connected layer를 삭제하여 layer가 13개가 되었고 하단에 Dense layer를 2개 사용했기 때문에 가중치, 편향 각각 개수 더했을 때 30이 된다.

#include top = False를 통해 shape를 고정?

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

