from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(datasets.DESCR)

# from tensorflow.keras.utils import to_categorical #케라스 (581012, 8)
# y = to_categorical(y) 

y = pd.get_dummies(y) #(581012, 7)

# y = y.reshape(-1,1) #(581012, 7)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

print(x.shape)
print(y.shape) #(581012, 7)

# print(pd.value_counts(y,))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, random_state=500,
                                                    stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# print(pd.value_counts(y_train,))

# #2. 모델 구성

# model = Sequential()
# model.add(Dense(128, activation= 'relu', input_dim = 54))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# #3 컴파일 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# start_time = time.time()

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience = 70,
#     restore_best_weights=True
#     )
# model.fit(x_train, y_train, epochs=1000, batch_size=300, verbose=1 , validation_split=0.25, callbacks=[es])
# end_time = time.time()

#평가 예측
model = load_model('./_save/keras30_mcp/10_fetch_cotype/k30_0726_2142_0088-0.1812.hdf5')
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss[0])
print("정확도 : ", round(loss[1], 3))

y_pred = model.predict(x_test)

print(y_pred)

# 로스값 :  0.22436301410198212
# 정확도 :  0.911

# 로스값 :  0.21188057959079742
# 정확도 :  0.918
# batch_size=200
# patience = 30

# 로스값 :  0.17798465490341187
# 정확도 :  0.934
# batch_size=300
# patience = 50

# minmaxscaler
# 로스값 :  0.14274589717388153
# 정확도 :  0.95

# standardscaler


# maxabscaler


# RobustScaler
# 로스값 :  0.13968294858932495
# 정확도 :  0.954
