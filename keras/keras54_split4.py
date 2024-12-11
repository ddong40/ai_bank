# (N, 10, 1) -> (N, 5, 2)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

print(x_predict)

size = 11


def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) #(91, 10)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x)
print(y)
print(x.shape, y.shape) #(90, 10) (90,)
# print(len(a)) 
x = x.reshape(
    x.shape[0], x.shape[1], 1)
x = x.reshape(90, 5, 2)
print(x.shape) #(90, 5, 2)

# print(x.shape) #(90, 10, 1)
# print(y.shape) #(90,)


model = Sequential()
model.add(LSTM(32, return_sequences= True, input_shape=(5,2)))
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
x_test = np.array(range(96, 106)).reshape(1, 5, 2)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('x_test의 결과 :', y_predict)

# 로스 :  0.05974710360169411
# x_test의 결과 : [[104.43384]]

# (n,5,2) 
# 로스 :  0.005453492980450392
# x_test의 결과 : [[103.92493]]