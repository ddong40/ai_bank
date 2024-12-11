import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping


a = np.array(range(1, 11))

size = 5


def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) #(6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) #(6, 4) (6,)
print(len(a)) 
x = x.reshape(
    x.shape[0], x.shape[1], 1)
print(x.shape) #(6, 4, 1)


model = Sequential()
model.add(LSTM(32, return_sequences= True, input_shape=(4,1)))
model.add(LSTM(32))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x, y, epochs=1550, batch_size=16, verbose=1, ) 

loss = model.evaluate(x, y)
x_test = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('[7, 8, 9, 10]의 결과 :', y_predict)

# 로스 :  7.235757493617712e-06
# [7, 8, 9, 10]의 결과 : [[10.868346]]