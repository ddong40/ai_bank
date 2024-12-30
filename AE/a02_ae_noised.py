import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf

np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.
# 평균 0 , 표편 0.1인 정규분포

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape) #(60000, 784) (10000, 784) # shape는 바뀌지 않는다.

print(np.max(x_train), np.min(x_test)) #1.0 0.0

print(np.max(x_train_noised), np.min(x_test_noised)) #1.506013411202829 -0.5281790150375157

x_train_noised = np.clip(x_train_noised,0,1)
x_test_noised = np.clip(x_test_noised,0,1)

print(np.max(x_train_noised), np.min(x_train_noised)) #1.0 0.0
print(np.max(x_test_noised), np.min(x_test_noised)) #1.0 0.0

#2. 모델

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784, ))
## 인코더
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)          # 히든레이어가 클 수록 잘 나온다.     
encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)     # 더 선명해짐. 별차이없다.
# encoded = Dense(2048, activation='relu')(input_img)


## 디코더
# decoded = Dense(784, activation='relu')(encoded)      
# decoded = Dense(784, activation='linear')(encoded)      # relu보다 뿌연게 생겼다. 음수땜시
decoded = Dense(784, activation='sigmoid')(encoded)       # relu랑 비슷, 조금 더 좋은거같아요
# decoded = Dense(784, activation='tanh')(encoded)          # 뿌연게 생겼다. 음수땜시


autoencoder = Model(input_img, decoded)

#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# relu, linear 랑은 별로. tanh도 별로.   sigmoid 랑 제일좋다.                       액티베이션과 로스 잘 생각해서 설정해줄것.

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()







