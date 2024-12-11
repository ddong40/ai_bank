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

# model.trainable = False
model.summary()

print(len(model.weights)) #layer당 길이가 2개가 나옴, 가중치와 bias 하나씩 계산했을 때 
print(len(model.trainable_weights)) # 4개인 이유 하단에 dense 2개는 훈련시킬 수 있기에  
'''
trainable에 따른 길이    Trainable : True     Trainable : False            vgg.trainable : False
len(model.weights)           30            //      30               //               30
len(model.trainable_w)       30            //      0                //                4
'''



