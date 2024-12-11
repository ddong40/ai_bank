from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, 
    #maxlen = 100,
    test_split=0.2,
)

print(x_train)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(y_train)

print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train))) #46

#8982개의 넘파이 안에 리스트가 들어있는데 리스트마다 길이가 다르다.

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) #87 56 #길이가 다르다. #pad sequence 해줘야함

print("뉴스 기사의 최대길이 : ", max(len(i) for i in x_train)) #2376
print("뉴스 기사의 최소길이 : ", min(len(i) for i in x_train)) #13
print("뉴스 기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #145.53

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences #순차적으로 패딩을 넣겠다.

#맹그러봐!!! (15, 5)


x_train = pad_sequences(x_train, padding="pre", maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding="pre", maxlen=100, truncating='pre')

print(x_train.shape)
print(y_train.shape)

# y 원핫하고 맹그러봐

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten, Bidirectional

model = Sequential()
model.add(Embedding(1000, 10, input_length=100)) #돌아가
model.add(Bidirectional(LSTM(64))) 
model.add(Dense(64))
model.add(Dense(46, activation='softmax'))   

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    patience=80,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=1, validation_split=0.2, callbacks = [es] )

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
# y_predict = model.predict(x_test)

# print(np.round(y_predict))

# loss : [2.366300106048584, 0.6086375713348389]

# loss : [1.3606605529785156, 0.6843277215957642]