import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(32, 32, 3))    

vgg16.trainable=True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()


####### [실습] ########
# 비교할것
# 1. 이전에 본인이 한 최상의 결과와
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=Ture,(디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, trainable=False 
#### 위에 2,3번할때는 time 체크할 것  

# 3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

# path1 = './_save/keras38/_cifa10/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = ''.join([path1, 'k30_', date, '_', filename])


# mcp = ModelCheckpoint(
#     monitor= 'val_loss',
#     mode = 'auto',
#     verbose=1,
#     save_best_only= True,
#     filepath = filepath
# )
    
# rlr = ReduceLROnPlateau(
#     monitor= 'val_loss',
#     mode = 'auto',
#     verbose = 0,
#     patience= 15,
#     factor= 0.9
# )

model.fit(x_train, y_train, epochs = 1, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
# y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

# acc = accuracy_score(y_test, y_predict)
print("로스값 : ", loss[0])
print("정확도 : ", loss[1])
print('----------------------------')

# 가중치 동결
# 로스값 :  2.6119139194488525
# 정확도 :  0.12890000641345978

# 가중치 훈련
