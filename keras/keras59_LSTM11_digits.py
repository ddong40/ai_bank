from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

x, y =  load_digits(return_X_y=True)

print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)
#1797개의 이미지가 있는데 8byte*8byte의 이미지를 읽기 편하게 64byte로 늘린 것이다. 

print(pd.value_counts(y, sort=False)) # 디폴트는 내림차순 # ascending=True
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

print(load_digits().DESCR)
print(x.shape)
print(y.shape)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle= True, 
                                                    random_state=100, stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(y.shape)

#2 모델 구성
model = Sequential()
model.add(Dense(128, activation= 'relu', input_dim=64))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights =False
    )
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras30_mcp/11_digits/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k30_', date, '_', filename])

mpc = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose =1,
    save_best_only =True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=1000, batch_size=2, verbose=1, validation_split=0.25, callbacks=[es, mpc])

end_time = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print("로스 : ", loss[0])
print("정확도 : ", round(loss[1], 3))
print("시간 : ", round(end_time - start_time, 2),'초')




# 로스 :  0.974841296672821
# 정확도 :  0.972
# 시간 :  61.33 초

# minmaxscaler
# 로스 :  0.35022199153900146
# 정확도 :  0.978

# standardscaler
# 로스 :  0.8481700420379639
# 정확도 :  0.963

# MaxAbsScaler


# RobustScaler
# 로스 :  0.7957172989845276
# 정확도 :  0.959

# 세이브 점수
# 로스 :  0.5483711361885071
# 정확도 :  0.959