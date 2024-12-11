import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

# 맹그러봐!!

token = Tokenizer() 
token.fit_on_texts([text1])
token.fit_on_texts([text2])

print(token.word_index) # {'마구': 1, '진짜': 2, '매우': 3, '못생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '태운이는': 11, '선생
#을': 12, '괴롭힌다': 13, '준영이는': 14, '사영이는': 15, '더': 16}

print(token.word_counts) 
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 6), ('먹었다', 1), ('태운 
# 이는', 1), ('선생을', 1), ('괴롭힌다', 1), ('준영이는', 1), ('못생겼다', 2), ('사영이는', 1), ('더', 1)])



x1 = token.texts_to_sequences([text1])
x2 = token.texts_to_sequences([text2])
print(x1)
print(x2)

x = np.concatenate((x1, x2), axis=1)

print(x)

# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x) #num_classes를 기존 컬럼보다 낮게 주면 안된다.
# x = x[:, :, 1:]
# x = np.reshape(x, (24, 16))
# print(x.shape)
# print(x)

# x = pd.get_dummies(np.array(x).reshape(-1,)) 
# print(x)
# print(x.shape)

#사이킷런 선생님꺼 
from sklearn.preprocessing import OneHotEncoder
x = np.reshape(x, (-1,1))
ohe = OneHotEncoder(sparse=False) #True가 디폴트
x = ohe.fit_transform(x)
print(x)
print(x.shape)