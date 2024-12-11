from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

#[실습] 맹그러봐
# R2 0.62 이상

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state= 8000)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=15)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어: ", r2)

# (x, y, train_size=0.9, random_state= 8000),  epochs=300, batch_size=15
# 로스 :  2360.693115234375
# r2 스코어:  0.604591142273077
# 로스 :  2228.980224609375
# r2 스코어:  0.6266526201768774