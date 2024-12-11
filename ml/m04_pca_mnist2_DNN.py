# m04_1에서 뽑은 4가지 결과로
# 4가지 모델을 맹그러
# input_shape = ()
# 1. 70000, 154
# 2. 70000, 331
# 3. 70000, 486

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)


# x =np.concatenate([x_train, x_test], axis=0)


# print(x.shape) #(70000, 28, 28)

x_train = x_train/255.
x_test = x_test/255.
# print(np.min(x), np.max(x)) #0.0 1.0

x_train = np.reshape(x_train, (60000, 28*28))
x_test = np.reshape(x_test, (10000, 28*28))

# pca = PCA(n_components=154)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

# print(x_train.shape)


##원핫 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(np.unique(y_train))

variable = np.array([154, 331, 486, 713, 748])

for i in reversed(variable):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
        
    #2 모델
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(i,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))

    #3 컴파일 훈련


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    start_time = time.time()

    es = EarlyStopping(
        monitor='val_loss',
        mode= 'min',
        verbose=1,
        patience=10,
        restore_best_weights=True
    )

    import datetime 
    date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
    print(date) #2024-07-26 16:49:51.174797
    print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
    print(date) #0726_1654
    print(type(date))


    path = './_save/keras38/_mnist/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
    filepath = "".join([path, 'k35_04', date, '_', filename])


    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only=True,
        filepath = filepath
    )

    model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.25, callbacks=[es, mcp])

    end_time = time.time()

    #4. 예측 평가
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    y_test1 = np.argmax(y_test).reshape(-1,1)
    y_predict1 = np.argmax(y_predict).reshape(-1,1)

    # acc = accuracy_score(y_test, y_predict)

    print('로스 : ', loss[0])
    print('정확도 :', loss[1])
    print('시간 ', round(end_time - start_time, 3), '초')

    # 713
    # 로스 :  0.022777607664465904
    # 정확도 : 0.9707000255584717
    # 시간  53.651 초

    # 486
    # 로스 :  0.02317901886999607
    # 정확도 : 0.96670001745224
    # 시간  36.935 초

    # 331
    # 로스 :  0.019483713433146477
    # 정확도 : 0.9739000201225281
    # 시간  49.637 초

