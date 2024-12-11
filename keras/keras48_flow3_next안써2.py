from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.2, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range=15, #각도 만큼 이미지 회전
    # zoom_range=1.2, #축소 또는 확대
    # shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest', #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  

augment_size = 100

print(x_train.shape) #(60000, 28, 28)
# print(x_train[0].shape) (28, 28)

# aaa = x_train[0].reshape(28, 28, 1)

# plt.imshow(aaa, cmap='gray')
# plt.show()

# aaa= np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
# print(aaa.shape)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=32, #디폴트
    shuffle=False,
)

print(xy_data)
print(type(xy_data))

# print(xy_data[0][0].shape) # (32, 28, 28, 1)
# print(xy_data[3][0].shape) # (4, 28, 28, 1)
# print(xy_data[4][0].shape) # 오류

# print(xy_data[0][1].shape) #(32,)

'''
print(len(xy_data)) #2

# print(xy_data[0].shape) 
print(xy_data[0][0].shape) #(100, 28, 28, 1)
#.next( ) 하기 전에는 앞에 한개의 것이 있는데 그것이 batch크기 
# .next( ) 하지 않으면 이터레이터 형태의 첫번째 배치를 가져온다. 
# 100, 28, 28, 1 을 32의 배치로 나눠 주겠다. 그러면 이터레이터는 4번이 반복될 것이다. 
# 32, 28, 28, 1이 세번, 그리고 4, 28, 28, 1이 생성됨. 
# 더 이상 쪼갤 양이 없기 때문에 다섯번 째 .next에서는 오류가 발생함.

# print(xy_data[1].shape) #(100,)

# #증폭시킨 데이터를 변환시킬 것이다. 

# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.imshow(xy_data[0][0][i], cmap='gray')
#     plt.axis('off')
# plt.show()
'''