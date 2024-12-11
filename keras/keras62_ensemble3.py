import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
# 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()
#원유, 환율, 금시세

x3_datasets = np.array([range(100), range(301, 401), 
                        range(77,177), range(33,133)]).T

# 거시기1, 거시기2, 거시기3, 거시기4

y1 = np.array(range(3001, 3101)) # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))

x3 = np.array([range(100,105), range(401,406)]).T
x4 = np.array([range(201, 206), range(511, 516), range(250, 255)]).T
x5 = np.array([range(100,105), range(401,406), range(177,182), range(133,138)]).T

print(x1_datasets.shape)
print(x2_datasets.shape)
print(x3_datasets.shape)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2, test_size=0.2, random_state=10, shuffle=True)


#2-1 모델구성
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='bit1')(input1)
dense2 = Dense(64, activation='relu', name='bit2')(dense1)
dense3 = Dense(128, activation='relu', name='bit3')(dense2)
dense4 = Dense(64, activation='relu', name='bit4')(dense3)
output1 = Dense(32, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)

# model1.summary()

#2-2 모델구성
input11 = Input(shape=(3,))
dense11 = Dense(32, activation='relu', name='bit11')(input11)
dense21 = Dense(64, activation='relu', name='bit21')(dense11)
dense31 = Dense(128, activation='relu', name='bit31')(dense21)
dense41 = Dense(64, activation='relu', name='bit41')(dense31)
output11 = Dense(32, activation='relu', name='bit51')(dense41)
# model2 = Model(inputs=input11, outputs=output11)

#2-3 모델구성
input22 = Input(shape=(4,))
dense22 = Dense(32, activation='relu', name='bit22')(input22)
dense32 = Dense(64, activation='relu', name='bit32')(dense22)
dense42 = Dense(128, activation='relu', name='bit33')(dense32)
dense52 = Dense(64, activation='relu', name='bit44')(dense42)
output22 = Dense(32, activation='relu', name='bit55')(dense52)

merge1 = Concatenate(name='mg1')([output1, output11, output22])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
last_output2 = Dense(1, name='last2')(merge3)

model = Model(inputs=[input1, input11, input22], outputs=[last_output, last_output2])

model.summary()


#컴파일 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras62/ensemble3/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1200, batch_size=16, verbose=1, validation_split=0.1, callbacks=[mcp])



loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
y_predict = model.predict([x3, x4, x5])

print(y_predict)
print(loss)

print(y_predict[0], y_predict[1])