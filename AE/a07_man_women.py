# keras45_7 ~ 등을 참고해서
# 남자 여자 사진에 noise를 주고, 내 사진도 노이즈를 만들고
# 오토인코더로 피부미백 훈련 가중치를 만든다.

# 그 가중치로 내 사진을 페딕트해서
# 피부미백 시킨다.

# 출력 이미지는 (원본, 노이즈, Predict) 순으로 출력

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL.Image as Image
#1. 데이터

test_datagen = ImageDataGenerator(
    rescale=1./255)

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/gender/'
x_train = np.load(np_path + 'keras43_01_x_train.npy')
# x_test = np.load(np_path + 'keras43_01_x_test.npy')

# x_train = x_train/255.
x_test = np.array(Image.open('./_data/_save_img/a.jpg').resize((100, 100))).reshape(1, 100, 100, 3) / 255.

x_train_noised = x_train[:5000] + np.random.normal(0, 0.1, size=x_train[:5000].shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)

x_train_noised = np.clip(x_train_noised,0,1)
x_test_noised = np.clip(x_test_noised,0,1)

x_train = x_train[:5000]

print(x_test_noised.shape)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, Flatten


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=100, kernel_size=(3,3), padding='same', input_shape= hidden_layer_size)) 
    model.add(MaxPool2D())
    model.add(Conv2D(50, kernel_size=(1,1), padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(50, kernel_size= (3,3), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(50, kernel_size=(1,1), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, padding='same', kernel_size=(3,3)))
    
    return model

hidden_size = (100, 100, 3)

model = autoencoder(hidden_size)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train_noised, x_train, epochs= 200, batch_size = 128, verbose=1)

decoded_imgs = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯개를 무작위로 고른다.
# random_images= random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[0])
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[0])
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[0])
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()
