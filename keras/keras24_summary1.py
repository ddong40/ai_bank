from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 4)                 16

#  dense_2 (Dense)             (None, 3)                 15        

#  dense_3 (Dense)             (None, 1)                 4

# Total params: 41
# Trainable params: 41 #훈련할 파라미터의 양
# Non-trainable params: 0 #훈련하지 않아도 될 파라미터의 양
#항상 뒤에 bias가 있다. 