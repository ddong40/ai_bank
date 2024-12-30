import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from sklearn.decomposition import PCA

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

# pca = PCA(n_components=1.0)

# x_train_noised = pca.fit_transform(x_train_noised)


#2. 모델

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape= (28*28,))) 
    model.add(Dense(784, activation= 'sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1) #
model_08 = autoencoder(hidden_layer_size=8) #
model_32 = autoencoder(hidden_layer_size=32) #
model_64 = autoencoder(hidden_layer_size=64) #
model_154 = autoencoder(hidden_layer_size=154) # PCA 1.0 
model_331 = autoencoder(hidden_layer_size=331) # PCA 99
model_486 = autoencoder(hidden_layer_size=486) # PCA 99.9
model_713 = autoencoder(hidden_layer_size=713) # PCA

#3. 컴파일, 훈련
print("===================== node 1개 시작 =========================")
model_01.compile(optimizer='adam', loss='mse')
model_01.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 8개 시작 =========================")
model_08.compile(optimizer='adam', loss='mse')
model_08.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 32개 시작 =========================")
model_32.compile(optimizer='adam', loss='mse')
model_32.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 64개 시작 =========================")
model_64.compile(optimizer='adam', loss='mse')
model_64.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 154개 시작 =========================")
model_154.compile(optimizer='adam', loss='mse')
model_154.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 331개 시작 =========================")
model_331.compile(optimizer='adam', loss='mse')
model_331.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 486개 시작 =========================")
model_486.compile(optimizer='adam', loss='mse')
model_486.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

print("===================== node 713개 시작 =========================")
model_713.compile(optimizer='adam', loss='mse')
model_713.fit(x_train_noised, x_train, epochs=10, batch_size=32, verbose=0)

#4. 평가, 예측
decoded_imgs_01 = model_01.predict(x_test_noised)
decoded_imgs_08 = model_08.predict(x_test_noised)
decoded_imgs_32 = model_32.predict(x_test_noised)
decoded_imgs_64 = model_64.predict(x_test_noised)
decoded_imgs_154 = model_154.predict(x_test_noised)
decoded_imgs_331 = model_331.predict(x_test_noised)
decoded_imgs_486 = model_486.predict(x_test_noised)
decoded_imgs_713 = model_713.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, axes  = plt.subplots(9, 5, figsize=(20, 7))

random_imgs= random.sample(range(decoded_imgs_01.shape[0]), 5)
outputs = [x_test, decoded_imgs_01, decoded_imgs_08, decoded_imgs_32, decoded_imgs_64,
           decoded_imgs_154, decoded_imgs_331, decoded_imgs_486, decoded_imgs_713]

# 원본(입력) 이미지를 맨 위에 그린다.
for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28),
                  cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.show()
    



