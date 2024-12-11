import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

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


# a = np.count_nonzero(x)
# print(a)
# b = np.count_nonzero(y)
# print(y)

# dict_a = {'0' : 0, '1' : 0} 
# for i in y:
#     if i == 0 :
#         dict_a['0'] += 1
#     else :
#         dict_a['1'] += 1
# print(dict_a)

# x_train, x_test, y_train, y_test = train_test_split(x, y,  )

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


#2. 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=30))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #accuracy와 acc는 같다.
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min', 
    patience = 30,
    restore_best_weights=True 
)
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


path1 = './_save/keras32/06_cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=1, validation_split = 0.2,
                 callbacks=[es, mcp]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 3))

y_pred = np.round(y_pred)


from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)
print("acc스코어 : ", accuracy_score)
print("걸린시간 : ", round(end - start, 2), "초" )



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

# 세이브 점수
# 로스 :  0.017171192914247513
# ACC :  0.971

# drop out
# 로스 :  0.009849203750491142
# ACC :  0.988