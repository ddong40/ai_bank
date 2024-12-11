import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import random as rn
rn.seed(1024)
tf.random.set_seed(1024)
np.random.seed(1024)


datagen = ImageDataGenerator(
    vertical_flip=False,
    horizontal_flip=True,
    fill_mode= 'nearest'
)

# path_train = 'C:/Users/ddong40/ai_2/_data/image/rps/'

# xy_train = train_datagen.flow_from_directory(
#     path_train,
#     target_size=(100,100),
#     batch_size=2520,
#     class_mode='categorical',
#     color_mode = 'rgb',
#     shuffle = True    
# )

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/rps/'
x_train = np.load(np_path + 'keras43_01_x_train.npy' )
y_train = np.load(np_path + 'keras43_01_y_train.npy')

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train))

augment_size = 7480

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)
print(y_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle= False
).next()[0]

print(x_augmented.shape)
print(y_augmented.shape)

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

print(x_train.shape) #(10000, 100, 100, 3)
# print(x_test.shape)
print(y_train.shape) #(2520, 3)



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=100)

print(x_train.shape) # (7000, 100, 100 , 3)
print(y_train.shape) # (7000, 3)


#모델
model = Sequential()
model.add(Conv2D(32, 2, input_shape=(100, 100, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr = np.array([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001])

for i in lr:
    

    #컴파일 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=i), metrics=['accuracy'])
    start_time = time.time()

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 20,
        restore_best_weights=True,
        verbose=0
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')

    path1 = './_save/keras41/rps/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path1, 'k30_', date, '_', filename])


    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        save_best_only=True,
        verbose=0,
        filepath=filepath
    )
    
    rlr = ReduceLROnPlateau(
        monitor = 'val_loss',
        mode = 'auto',
        verbose=0,
        patience=1,
        factor= 0.9
    )

    model.fit(x_train, y_train, epochs= 1, batch_size=10, verbose=0, validation_split=0.25, callbacks=[es, mcp, rlr])
    end_time = time.time()

    #평가 예측
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # y_test = np.argmax(y_test).reshape(-1, 1)
    # y_predict = np.argmax(y_predict).reshape(-1, 1)

    # acc = accuracy_score(y_test, y_predict)

    print('{0} : '.format(i))
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')
    # print('시간 :', round(end_time - start_time, 3), '초')

# 로스 :  0.006339925806969404
# 정확도 :  1.0
# 시간 : 31.643 초



# # model = load_model('_save/keras41/rps/k30_0802_1615_0019-0.0127.hdf5')
# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)

# y_test = np.argmax(y_test).reshape(-1, 1)
# y_predict = np.argmax(y_predict).reshape(-1, 1)

# acc = accuracy_score(y_test, y_predict)

# print('로스 : ', loss[0])
# print('정확도 : ', loss[1])
