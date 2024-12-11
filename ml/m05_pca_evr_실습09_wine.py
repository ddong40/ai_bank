import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_wine
import time

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(datasets)
print(datasets.DESCR)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

print(x.shape) #(178, 13)

from sklearn.decomposition import PCA
# pca = PCA(n_components=13)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# # print(np.argmax(cumsum >=1.0)+1) #13
# # print(np.argmax(cumsum >=0.999)+1) #2
# # print(np.argmax(cumsum >=0.99)+1) #1
# # print(np.argmax(cumsum >=0.95)+1) #1

list_a = np.array([1, 2, 13])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, 
                                                    random_state= 150, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    

    #2 모델구성

    model = Sequential()
    model.add(Dense(128, 'relu', input_dim=i))
    model.add(Dropout(0.3))
    model.add(Dense(128, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, 'softmax'))

    #컴파일 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    start_time = time.time()

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 50,
        restore_best_weights= True)

    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path1 = './_save/keras32/09_wine/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = "".join([path1, 'k30_', date, '_', filename])

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose =0,
        save_best_only=True,
        filepath = filepath
    )


    model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=0, validation_split=0.25, callbacks=[es, mcp])
    end_time = time.time()

    #평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("로스값 : ", loss[0])
    print("accuracy : ", round(loss[1],3))
    print("걸린시간 : ", round(end_time - start_time, 2), "초" )
    y_pred = model.predict(x_test)
    

# 로스값 :  0.2663238048553467
# accuracy :  0.907

# minmaxscaler
# 로스값 :  0.2635650038719177
# accuracy :  0.963

# standardscaler
# 로스값 :  0.21828098595142365
# accuracy :  0.981

# MaxAbsScaler
# 로스값 :  0.9603663682937622
# accuracy :  0.944

# RobustScaler
# 로스값 :  0.23232993483543396
# accuracy :  0.963

# 세이브 값
# 로스값 :  0.3106008470058441
# accuracy :  0.963

# drop out
# 로스값 :  0.4955110251903534
# accuracy :  0.944

# 13 
# 로스값 :  0.5735413432121277
# accuracy :  0.963
# 걸린시간 :  6.51 초

# 2 ★★★
# 로스값 :  0.21953085064888
# accuracy :  0.926
# 걸린시간 :  5.23 초

# 1
# 로스값 :  0.5383672714233398
# accuracy :  0.722
# 걸린시간 :  9.34 초