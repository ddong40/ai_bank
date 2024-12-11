import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)
# # ## test 이미지 파일명 변경 ##
import os
import natsort

file_path = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test"
file_names = natsort.natsorted(os.listdir(file_path))

print(np.unique(file_names))
i = 1
for name in file_names:
     src = os.path.join(file_path,name)
     dst = str(i).zfill(5)+ '.jpg'
     dst = os.path.join(file_path, dst)
     os.rename(src, dst)
     i += 1


start1 = time.time()


path_train = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"



np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

from tensorflow.keras.applications import VGG16

vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(100, 100, 3))    

# vgg16.trainable=False
vgg16.trainable=True


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

####### [실습] ########
# 비교할것
# 1. 이전에 본인이 한 최상의 결과와
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=Ture,(디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, trainable=False 
#### 위에 2,3번할때는 time 체크할 것 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras42/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k42_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
# print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
acc = accuracy_score(y_test, y_pre)
print('accuracy_score :', acc)


#False
# loss : 1.6031460762023926
# acc : 0.47608