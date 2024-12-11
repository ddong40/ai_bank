#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from sklearn.decomposition import PCA


#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)

# x = np.concatenate([x_train, x_test], axis=0)

# print(x.shape) #(506, 13)


pca = PCA(n_components=13)
x = pca.fit_transform(x)

########변화율 확인###########

# cunsum = np.cumsum(pca.explained_variance_ratio_)
# print(np.argmax(cunsum >=1.0)+1) #13
# print(np.argmax(cunsum >=0.999)+1) #6
# print(np.argmax(cunsum >=0.99)+1) #3
# print(np.argmax(cunsum >=0.95)+1) #2



from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) #2줄을 한 줄로 줄일 수 있다. 
x_test = scaler.transform(x_test)

list_a = np.array([2, 3, 6, 13]) 

for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)


    #2. 모델구성
    model = Sequential()
    # model.add(Dense(100, input_dim=13)) 
    model.add(Dense(64, input_shape=(i,))) #백터형태로 받아들인다 백터가 어차피 column 이니까 #이미지일때는 input_shape=(8,8,1) 
    model.add(Dropout(0.3)) #30 percenet를 빼서 훈련을 시키지 않는다. 상위 레이어에서 drop out한다는 뜻
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3)) # 드롭아웃은 통상 0.5까지
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1)) 
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))




    #3. 컴파일, 훈련

    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 10,
        verbose = 1,
        restore_best_weights=True
    )

    #################mcp 세이브 파일명 만들기 시작##################

    import datetime 
    date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
    print(date) #2024-07-26 16:49:51.174797
    print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
    print(date) #0726_1654
    print(type(date))


    path = './_save/keras32/01_boston'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
    filepath = "".join([path, 'k32_', date, '_', filename])

    ############mcp세이브 파일명 만들기 끝################

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 1, #가장 좋은 지점을 알려줄 수 있게 출력함
        save_best_only=True,
        filepath = filepath
    )
    # 생성 예" './_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5'


    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split = 0.3, callbacks=[es, mcp])
    end = time.time()

    # model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test) 

    y_predict = model.predict(x_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_predict)

    print('로스 : ', loss)
    print('r2 score :', r2)

    # 로스 :  0.2362709939479828
    # 정확도 :  0.913

    # RobustScaler
    # 로스 :  17.011682510375977
    # r2 score : 0.8183080232863937


    # 로스 :  8.781745910644531
    # r2 score : 0.9062072413873566

    # Epoch 00214: val_loss did not improve from 13.15108

    # drop out..
    # 로스 :  15.432994842529297
    # r2 score : 0.8351691183396657
    
    
    
# 13   ★★★
# 로스 :  12.33582878112793
# r2 score : 0.8682481618376505


# 6
# 로스 :  18.08567237854004
# r2 score : 0.8068373919796679
    
# 3   
# 로스 :  59.77436447143555
# r2 score : 0.36158462792555557   
    
    
# 2    
# 로스 :  72.95055389404297
# r2 score : 0.22085739627870626