import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
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

x_test = ['태운이 참 재미없다']
x_test = token.texts_to_sequences(x_test)
print(x_test)
x_test = pad_sequences(x_test, padding="pre", maxlen=5)


# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)

# print(x)

# print(x.shape) #(15, 5)

# x = x[:, :, 1:]

print(x)
print(x.shape) #(15, 5, 31) 

#2 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
###################### 임베딩1 ##############################
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5)) #(None, 5, 100)

#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 7,661

#####################임베딩2#################################
# model.add(Embedding(input_dim=31, output_dim=100))

#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 7,661

# ###################임베딩3####################################
# model.add(Embedding(input_dim=100, output_dim=100))

# input_dim이 단어사전의 갯수보다 적을 때 : 연산량 줄어, 단어사전에서 임의로 뺀다. 성능 조금 저하
# input_dim이 단어사전의 갯수보다 클 때 : 연산량 늘어, 임의의 랜덤 임베딩 생성. 성능 조금 저하

###################임베딩4####################################
model.add(Embedding(31, 100)) #돌아가
# model.add(Embedding(31, 100, 5)) #안돌아가
# model.add(Embedding(31, 100, 1)) #돌아가, input_lenth의 약수로 돌아간다. 

model.add(LSTM(10)) #(None, 10)
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))   
 
model.summary()

exit()

#3 컴파일 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x, labels, epochs=100, batch_size=16)

#4 평가 예측
loss = model.evaluate(x, labels)
print('loss :', loss)
y_predict = model.predict(x_test)

print(np.round(y_predict))

