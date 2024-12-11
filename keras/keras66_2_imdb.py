from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000
)

print(x_train)
print(x_train.shape, x_test.shape) #25000 #25000
print(y_train.shape, y_test.shape) 
print(y_train)
print(np.unique(y_train)) # 0 1

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) #218 #189

print("imdb의 최대길이 : ", max(len(i) for i in x_train)) #2494
print("imdb의 최소길이 : ", min(len(i) for i in x_train)) #11
print("imdb의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #238.71364

from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding="pre", maxlen=200, truncating='pre')
x_test = pad_sequences(x_test, padding="pre", maxlen=200, truncating='pre')

print(x_train.shape)
print(y_train.shape)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)






from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten, Bidirectional

model = Sequential()
model.add(Embedding(1000, 10)) #돌아가
model.add(Bidirectional(LSTM(64))) 
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))   

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=1, validation_split=0.2, callbacks = [es] )

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# loss : [0.3360758125782013, 0.855239987373352]