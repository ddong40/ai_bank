# a3_ae2를 카피해서 모델 구성

# 모든 패딩 same

# 인코더 28
# conv 28
# maxpool 14
# conv 14
# maxpool 7

#디코더 
# conv 7
# upsampling 2d(2,2) 14
# conv 14 
# upsampling 2d(2,2) 28
# conv(1, (3,3)) -> (28, 28, 1) 로 맹그러

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from sklearn.decomposition import PCA

np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
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
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, Flatten


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same', input_shape= hidden_layer_size)) 
    model.add(MaxPool2D())
    model.add(Conv2D(10, kernel_size=(1,1), padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(10, kernel_size= (3,3), padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(10, kernel_size=(1,1), padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1, padding='same', kernel_size=(3,3)))
    
    return model

hidden_layer = (28, 28, 1)

model = autoencoder(hidden_layer)

model.compile(optimizer ='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=10, verbose=1, batch_size = 32)

decoded_imgs = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯개를 무작위로 고른다.
random_images= random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()