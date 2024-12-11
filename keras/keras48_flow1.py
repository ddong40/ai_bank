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

###################### 요기부터 증폭 #############################

datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True, # 수평 뒤집기
    # vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.2, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range=15, #각도 만큼 이미지 회전
    # zoom_range=1.2, #축소 또는 확대
    # shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest', #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  

it = datagen.flow(img,
             batch_size=1) #flow from directory는 이미지를 가져다가 증폭하는데, flow는 수치화 된 데이터를 가져다가 증폭
#flow는 이미 수치화 되어있어서 target, 경로, color, shuffle 필요 없음. batch는 필요함
# it는 이터레이터 형태이다. 프린트해보면 
print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5)) #subplots는 그림 여러장을 연속적으로 1행 5열으로 그림 뽑겠다. figsize는 그림판의 사이즈

for i in range(5):
    batch = it.next() #안에 놈 보고 싶기 때문에 5번 next 함수 반복하여 봐줌, 
    print(batch.shape) #(1, 100, 100, 3)
    batch = batch.reshape(100, 100, 3)
    
    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()



# np_path = 'C:/Users/ddong40/ai_2/_data/image/me/'
# np.save(np_path + 'me.npy', arr=img)
# # np.save(np_path + 'keras45_me.npy', arr[0][1])