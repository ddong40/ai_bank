import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)


#1 .데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']


print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]


x = x[:-39]
y = y[:-39]

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

print(y)

# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]

print(np.unique(y, return_counts=True))

# (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25, shuffle=True,  random_state=123, stratify=y)

#2 모델
'''
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))



#3 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 평가, 예측
results = model.evaluate(x_test, y_test)
print( 'loss : ', results[0])
print('acc : ', results[1])


# 지표 : f1_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print(y_predict)


acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')


print('acc : ', acc)
print('f1_score : ', f1)
'''
# acc :  0.8571428571428571
# f1_score :  0.5970961887477314



##################### SMOTE 적용 ###############################
#pip install imblearn

from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk

print('사이킷런 : ', sk.__version__) #사이킷런 :  1.1.3

print(np.unique(y_train, return_counts = True))
# (array([0, 1, 2]), array([44, 53,  6], dtype=int64))

# smote = SMOTE(random_state=7777)
ros = RandomOverSampler(random_state=7777)
x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts = True))
# (array([0, 1, 2]), array([53, 53, 53], dtype=int64))

################스모팅 적용 끝###############


model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))



#3 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 평가, 예측
results = model.evaluate(x_test, y_test)
print( 'loss : ', results[0])
print('acc : ', results[1])


# 지표 : f1_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print(y_predict)


acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')


print('acc : ', acc)
print('f1_score : ', f1)

# smote
# acc :  0.8857142857142857
# f1_score :  0.6259259259259259

# ros 적용 후 #범주형에서 잘 먹힌다는 이야기가 있음
# acc :  0.8857142857142857
# f1_score :  0.6259259259259259