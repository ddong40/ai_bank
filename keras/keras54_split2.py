import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T
print(a.shape) #(10, 2)

size = 6 
print(a)

def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + (size))]
        aaa.append(subset)
    return np.array (aaa)

bbb = split_x(a, size)
print('---------------------')
print(bbb)
print(bbb.shape) #(5, 6, 2)

x = bbb[:, :-1]
y = bbb[:, -1, 0]
print(x)
print(y)
print(x, y)
print(x.shape, y.shape) #(5, 5, 2) (5,)

'''
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
x_test = np.array([[6,4], [7,3], [8,2], [9,1], [10,0]]).reshape(1, 5, 2)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('[6,4], [7,3], [8,2], [9,1], [10,0]의 결과 :', y_predict)
'''