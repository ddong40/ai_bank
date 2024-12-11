import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #색상 값이 숨겨져있다. 사실은 60000, 28, 28, 1
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
    #   dtype=int64))
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892
# dtype: int64
# 마지막 layer는 softmax, output 10


import matplotlib.pyplot as plt
plt.imshow(x_train[59999], 'gray') #xtrain의 59999번째를 보여주겠다 , 'gray' 색상을 흑백으로 하겠다. 
plt.show()
print('y_train[59999]의 값 : ', y_train[59999])