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



# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 이상 #154개
# 0.99 이상 #331개
# 0.999 이상 #486개
# 1.0 일때 몇개? #713개


# hidden_size = 713   # PCA가 1.0일 때 # loss: 0.0635 - val_loss: 0.0642
# hidden_size = 486 # PCA가 0.999일 때 # loss: 0.0635 - val_loss: 0.0642
# hidden_size = 331 # PCA가 0.99일 때 # loss: 0.0636 - val_loss: 0.0643
# hidden_size = 154 # PCA가 0.95일 때 # loss: 0.0647 - val_loss: 0.0654

hidden_size = [128, 64, 32, 64, 128]
# hidden_layer_size = [64, 128, 256, 128, 64]
# hidden_layer_size = [128, 128, 128, 128, 128]
# 위 세개 비교 / 모레시계형, 다이아몬드형, 통나무형 / 이름도 잘 지었다.

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size[0], input_shape=(784,)))
    model.add(Dense(units=hidden_layer_size[1]))
    model.add(Dense(units=hidden_layer_size[2]))
    model.add(Dense(units=hidden_layer_size[3]))
    model.add(Dense(units=hidden_layer_size[4]))
    model.add(Dense(784, activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=hidden_size)    


#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='binary_crossentropy')
# relu, linear 랑은 별로. tanh도 별로.   sigmoid 랑 제일좋다.                       액티베이션과 로스 잘 생각해서 설정해줄것.

model.fit(x_train, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

# import matplotlib.pyplot as plt
# n = 10
# plt.figure(figsize=(20,4))
# for i in range(n):
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test_noised[i].reshape(28,28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_xaxis().set_visible(False)

#     ax = plt.subplot(2, n, i+1+n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()


##########################

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

# loss: 0.0922 - val_loss: 0.0931
