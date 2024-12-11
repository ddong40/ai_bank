import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=5, #각도 만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest', #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  
 
# 이 데이터들은 훈련에서 사용될 것 들이다.

test_datagen = ImageDataGenerator(
    rescale=1./255
) #평가 데이터이기 때문에 rescale 외에 다른 변환을 하지 않는다.

path_train = './_data/image/brain/train/' #라벨이 분류된 상위 폴더까지 path를 잡는다. 이후 수치화하면 0과 1로 바뀐다.
path_test = './_data/image/brain/test/' 

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True) #imagedatagenertor방식으로 디렉토리부터 흘러와서 이미지를 수치화하여 xy_train에 담아라
# batch를 10으로 주면 16*(10, 200, 200, 1)
# train폴더에 ad와 normal이 각각 80개 

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False) #test데이터는 suffle을 할 필요가 없다.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000015256384F40>
print(xy_train.next()) #iterator의 첫 번째 데이터를 보여줘 # array([0., 0., 1., 0., 0., 1., 0., 0., 1., 1.], dtype=float32)) 
print(xy_train.next()) #array([0., 0., 1., 1., 0., 0., 0., 0., 0., 0.], dtype=float32))

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])

# print(xy_train[0].shape)
# AttributeError: 'tuple' object has no attribute 'shape'
print(xy_train[0][0].shape) #16개 #2개?
# (10, 200, 200, 1)

# print(xy_train[16]) 
# ValueError: Asked to retrieve element 16, but the Sequence has length 16

# print(xy_train[15][2])
# IndexError: tuple index out of range

print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

