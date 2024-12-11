import sklearn as sk
import numpy as np
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      #(506, 13)    --> input_dim=13
print(y)
print(y.shape)      #(506,)  벡터

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=380, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print('RMSE :', rmse)


# # (x, y, train_size=0.8, random_state=3333)  epochs=400, batch_size=5
# # 로스 :  19.67884063720703
# # r2스코어 :  0.7459871274196113
# # (x, y, train_size=0.8, random_state=6666)  epochs=400, batch_size=5
# # 로스 :  20.507410049438477
# # r2스코어 :  0.7809722401357289
# # (x, y, train_size=0.8, random_state=6666)  epochs=500, batch_size=5
# # 로스 :  20.05512809753418
# # r2스코어 :  0.7858027954982522
