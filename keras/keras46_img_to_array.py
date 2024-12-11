import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
import matplotlib.pyplot as plt

path = 'C:/Users/ddong40/ai_2/_data/image/me/me.jpg'

img = load_img(path, target_size=(100,100),)
print(img) #<PIL.Image.Image image mode=RGB size=200x200 at 0x202B203A6A0>

# print(type(img))
# plt.imshow(img)
# plt.show() # 잘생긴 내 사진을 보았다.

arr = img_to_array(img)
# print(arr)
# print(arr.shape) #(146, 180, 3) -> (100, 100, 3)
# print(type(arr)) #(100, 100, 3) -> reshape로 4차원으로 바꿔줘야함! 

#차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape) #(1, 100, 100, 3)

# me 폴더에 위에 데이터를 npy로 저장할 것



np_path = 'C:/Users/ddong40/ai_2/_data/image/me/'
np.save(np_path + 'me.npy', arr=img)
# np.save(np_path + 'keras45_me.npy', arr[0][1])