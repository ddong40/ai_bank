import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC



#1 데이터 data 
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)
print(type(x)) #<class 'numpy.ndarray'>

print(np.unique(y, return_counts=True))

#
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

print(x.shape) # (569, 30)

n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 모델
model = SVC()

#3 훈련

scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n평균 ACC :', round(np.mean(scores),4))

#kfold
# ACC :  [0.92105263 0.87719298 0.90350877 0.94736842 0.91150442] 
# 평균 ACC : 0.9121

#starifiedkfold
# ACC :  [0.92105263 0.93859649 0.92105263 0.92982456 0.86725664] 
# 평균 ACC : 0.9156

exit()
from sklearn.decomposition import PCA

pca = PCA(n_components=30)
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(np.argmax(cumsum >= 1.0)+1) #1
print(np.argmax(cumsum >= 0.999) +1) #3
print(np.argmax(cumsum >= 0.99) +1) #2
print(np.argmax(cumsum >= 0.95) +1) #1

list_a = np.array([1, 2, 3, 30])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=500)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

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

    #2. 모델구성
    model = Sequential()
    model.add(Dense(64,activation='relu', input_dim=i))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['acc']) #accuracy와 acc는 같다.
    start = time.time()

    from tensorflow.keras.callbacks import EarlyStopping 
    es = EarlyStopping(
        monitor='val_loss', 
        mode = 'min', 
        patience = 30,
        restore_best_weights=True 
    )

    hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=0, validation_split = 0.2,
                     callbacks=[es]
                     )
    end = time.time()

    #4. 평가, 예측
    # model = load_model('./_save/keras30_mcp/06_cancer/k30_0726_2044_0016-0.0292.hdf5')
    loss = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    print("로스 : ", loss[0])
    print("ACC : ", round(loss[1], 3))

    y_pred = np.round(y_pred)


    from sklearn.metrics import r2_score, accuracy_score
    accuracy_score = accuracy_score(y_test, y_pred)
    print("acc스코어 : ", accuracy_score)
    # print("걸린시간 : ", round(end - start, 2), "초" )



    #로스 :  0.4035087823867798
    #ACC :  0.596


    # 로스 :  0.34502923488616943
    # ACC :  0.655

    # 로스 :  0.04576372727751732
    # ACC :  0.953 

    # minmaxscaler
    # 로스 :  0.030928198248147964
    # ACC :  0.959

    # StandardScaler
    # 로스 :  0.03041931986808777
    # ACC :  0.965

    # maxabsscaler
    # 로스 :  0.0314495787024498
    # ACC :  0.959

    # RobustScaler
    # 로스 :  0.010275790467858315
    # ACC :  0.988

    #load data
    # 로스 :  0.017171192914247513
    # ACC :  0.971
    

# 1 ★★★ 
# 로스 :  0.05663470923900604
# ACC :  0.918
# acc스코어 :  0.9181286549707602

# 2
# 로스 :  0.1310318410396576
# ACC :  0.825
# acc스코어 :  0.8245614035087719

# 3
# 로스 :  0.14942196011543274
# ACC :  0.784
# acc스코어 :  0.783625730994152

# 30
# 로스 :  0.1382851004600525
# ACC :  0.836
# acc스코어 :  0.8362573099415205