
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

#1 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다'
]

# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, 
# '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x) 
print(type(x)) #<class 'list'>


from tensorflow.keras.preprocessing.sequence import pad_sequences #순차적으로 패딩을 넣겠다.

#맹그러봐!!! (15, 5)


x = pad_sequences(x, padding="pre")

print(x)

from tensorflow.keras.utils import to_categorical
x = to_categorical(x)

print(x)

print(x.shape) #(15, 5)

# x = x[:, :, 1:]

print(x)
print(x.shape)
 #num_classes를 기존 컬럼보다 낮게 주면 안된다.

# x = np.reshape(x, (24, 16))

x_test = ['태운이 참 재미없다']

# token.fit_on_texts(x_test)
x_test = token.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, padding = 'pre',  maxlen=5)
x_test = to_categorical(x_test, num_classes = 31).reshape(1, 5, 31)

print(x_test)


# x_test = x_test[:, :, 1:]


# print(x_test)

# x_test2 = ['태운이 참 최고에요']
# x_test2 = token.texts_to_sequences(x_test2)
# x_test2 = pad_sequences(x_test2, maxlen=5)

# print(x_test2)

# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(5, 31)))
model.add(Dense(32, activation='relu' ))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])

# metrics=['accuracy']

model.fit(x, labels, epochs=500, batch_size=16)

loss = model.evaluate(x, labels)
y_predict = model.predict(x_test)
# acc = accuracy_score(labels, y_predict)

print(loss)
print(np.round(y_predict))
# print(acc)

# 0.4843558371067047
# [[0.]]
