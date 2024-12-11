from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

x, y =  load_digits(return_X_y=True)

from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

model = xgb.XGBClassifier()

scores  = cross_val_score(model, x_train, y_train, cv=kfold) #cv -> cross validation
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

acc = accuracy_score(y_test, y_predict)

print('cross_val_predict ACC :', acc)
#kfold
# ACC :  [0.94444444 0.95833333 0.96935933 0.98885794 0.95821727] 
# 평균 ACC :  0.9638

#starified kfold
# ACC :  [0.96944444 0.96666667 0.94986072 0.97214485 0.96100279] 
# 평균 ACC :  0.9638

# ACC :  [0.93055556 0.96180556 0.96167247 0.95818815 0.95470383] 
# 평균 ACC :  0.9534
# cross_val_predict ACC : 0.8833333333333333

exit()
print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)
#1797개의 이미지가 있는데 8byte*8byte의 이미지를 읽기 편하게 64byte로 늘린 것이다. 

from sklearn.decomposition import PCA

pca = PCA(n_components=64)
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(np.argmax(cumsum >= 1.0)+1) #61
print(np.argmax(cumsum >= 0.999)+1) #49
print(np.argmax(cumsum >= 0.99)+1) #41
print(np.argmax(cumsum >= 0.95)+1) #29


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

list_a = np.array([29, 41, 49, 61, 64])

#2 모델 구성
for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    model = Sequential()
    model.add(Dense(128, activation= 'relu', input_dim=64))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.2))
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

    path1 = './_save/keras32/11_digits/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = "".join([path1, 'k30_', date, '_', filename])

    mpc = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose =0,
        save_best_only =True,
        filepath = filepath
    )


    model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=0, validation_split=0.25, callbacks=[es, mpc])

    end_time = time.time()

    #4 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)

    print("로스 : ", loss[0])
    print("정확도 : ", round(loss[1], 3))
    print("시간 : ", round(end_time - start_time, 2),'초')
    print('-----------------------------------------------')



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

# drop out
# 로스 :  0.23905232548713684
# 정확도 :  0.969