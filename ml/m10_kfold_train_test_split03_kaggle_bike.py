# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score
import time 
from sklearn.preprocessing import StandardScaler

#1. 데이터
path = 'C:/Users/ddong40/ai_2/_data/kaggle/bike-sharing-demand/' #절대경로(경로가 풀로 다 들어간 경우)
# path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand' #위와 다 같음
# path = 'C://Users//ddong40//ai//_data//bike-sharing-demand' #위와 다 같음

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) #(6493, 1)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object'
    
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())

######### 결측치 확인 ###########

print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())


###### x와 y분리
x = train_csv.drop(['casual','registered','count'], axis=1) #이 리스트의 컬럼들을 axis 1에 넣어 드랍해주세요 라는 뜻
print(x)
y = train_csv['count']

print(x.shape) #(10886, 8)
print(test_csv.shape) #(6493, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#kfold

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#모델 
model = SVR()

#훈련 
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict ACC :', r2)



# 8
# 로스 :  22293.607421875 ★
# r2 score : 0.3101296626105775

# kfold
# ACC :  [0.20161519 0.18059705 0.21608856 0.21685107 0.17728516] 
# 평균 ACC :  0.1985

# train_split 이후
# ACC :  [0.18651857 0.20414629 0.20375396 0.20431597 0.23228958] 
# 평균 ACC :  0.2062
# cross_val_predict ACC : 0.12963650209543787
 

exit()




from sklearn.decomposition import PCA

# pca = PCA(n_components=8)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(np.argmax(cumsum >= 1.0)+1) #8
# print(np.argmax(cumsum >= 0.999)+1) #5
# print(np.argmax(cumsum >= 0.99)+1) #3
# print(np.argmax(cumsum >= 0.95)+1) #3

list_a = np.array([3, 5, 8])





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=100)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()    
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    test_csv = pca.transform(test_csv)



    #2 모델구성
    model = Sequential()
    model.add(Dense(256, activation= 'relu', input_dim = i))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(1, activation= 'linear'))

    #3 컴파일 훈련
    model.compile(loss = 'mse', optimizer='adam')
    start = time.time()

    from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience = 10,
        restore_best_weights = True 
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")
    path1 = './_save/keras32/05_kaggle_bike/'
    filename = '{epoch:04d}-{val_loss:4f}.hdf5'
    filepath = "".join([path1, 'k30_', date, '_', filename])

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 1,
        save_best_only= True,
        filepath = filepath    
    )

    model.fit(x_train, y_train, epochs=500, batch_size=10, verbose=0, validation_split=0.25,
                    callbacks=[es, mcp])
    end = time.time()

    #4 평가 예측
    loss = model.evaluate(x_test, y_test, verbose = 0)
    y_predict = model.predict(x_test)
    y_submit = model.predict(test_csv)
    r2 = r2_score(y_test, y_predict)

    #5. 파일 출력
    sampleSubmission['count'] = y_submit
    print(sampleSubmission)

    sampleSubmission.to_csv(path+'samplesubmission_0725_1429.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임
    print('로스 : ', loss)
    print('r2 score :', r2)

# print(hist)
# print('======================= hist.history==================')
# print(hist.history)
# print('================loss=================')
# print(hist.history['loss'])
# print('=================val_loss==============')
# print(hist.history['val_loss'])
# print('====================================================')


# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right') #라벨 값이 무엇인지 명시해주는 것이 레전드
# plt.title('캐글 Loss') #그래프의 제목 
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# 스케일링 안 했을 때
# 로스 :  22878.80859375
# r2 score : 0.2920206412513767

# minmaxscaler
# 로스 :  21880.876953125
# r2 score : 0.3229013736943528

# StandardScaler
# 로스 :  21804.029296875
# r2 score : 0.3252792516669306

# maxabsscaler
# 로스 :  21630.7265625
# r2 score : 0.3306426081924827

# RobustScaler
# 로스 :  21642.716796875
# r2 score : 0.33027123377950784

# 세이브 점수
# 로스 :  21703.53125
# r2 score : 0.3283893159099268

# drop out
# 로스 :  22320.208984375
# r2 score : 0.3093064548441353


# 8
# 로스 :  22293.607421875 ★
# r2 score : 0.3101296626105775

# 5
# 로스 :  23368.50390625
# r2 score : 0.27686722823994525

# 3
# 로스 :  24025.814453125
# r2 score : 0.2565267009537232