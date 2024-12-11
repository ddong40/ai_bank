import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=1050,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(
            xy_train[0][0],
            xy_train[0][1],
            train_size=0.8,
            shuffle=False
)


from tensorflow.keras.applications import VGG16

vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(150, 150, 3))    

vgg16.trainable=True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=30,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)

####### [실습] ########
# 비교할것
# 1. 이전에 본인이 한 최상의 결과와
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=Ture,(디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, trainable=False 
#### 위에 2,3번할때는 time 체크할 것  



# loss : 0.0007545964326709509
# acc : 1.0
# acc_score : 1.0

# False
# loss : 0.00303265661932528
# acc : 0.5291
# acc_score : 0.470873786407767

# True
# loss : 0.015368040651082993
# acc : 0.5291
# acc_score : 0.470873786407767

###### batch_size 변경후 성능 향상!!!!!!!



