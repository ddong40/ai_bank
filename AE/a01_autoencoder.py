import numpy as np
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784, ))
## 인코더
encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)          # 히든레이어가 클 수록 잘 나온다.     
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)     # 더 선명해짐. 별차이없다.
# encoded = Dense(2048, activation='relu')(input_img)


## 디코더
# decoded = Dense(784, activation='relu')(encoded)      
# decoded = Dense(784, activation='linear')(encoded)      # relu보다 뿌연게 생겼다. 음수땜시
# decoded = Dense(784, activation='sigmoid')(encoded)       # relu랑 비슷, 조금 더 좋은거같아요
decoded = Dense(784, activation='tanh')(encoded)          # 뿌연게 생겼다. 음수땜시


autoencoder = Model(input_img, decoded)

#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# relu, linear 랑은 별로. tanh도 별로.   sigmoid 랑 제일좋다.                       액티베이션과 로스 잘 생각해서 설정해줄것.

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 인코더 5개의 히든과
# 디코드 4개의 엑티베이션
# 그리고 2개의 컴파일-로스 부분의 경우의 수
# 총 40가지의 로스를 정리하고
# 눈으로 결과치를 비교해 볼것 // 뭐가 젤 좋은지???
