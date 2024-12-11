import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:/Users/ddong40/ai_2/_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=2520,
    class_mode='binary',
    color_mode = 'rgb',
    shuffle = True    
)

# xy_test = test_datagen.flow_from_directory(
#     path_train,
#     target_size= (100,100)
#     batch_size=2520,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True
# )

x = xy_train[0][0]
y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=100)


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
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True,
    verbose=1
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras41/horse/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath=filepath
)

model.fit(x_train, y_train, epochs= 300, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test).reshape(-1, 1)
y_predict = np.argmax(y_predict).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('정확도 : ', loss[1])
print('시간 :', round(end_time - start_time, 3), '초')

# 로스 :  0.024591155350208282
# 정확도 :  0.9902912378311157
# 시간 : 12.092 초

# 로스 :  0.026476573199033737
# 정확도 :  0.9935275316238403
# 시간 : 39.311 초