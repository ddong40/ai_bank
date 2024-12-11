import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
              [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predcit = np.array([50,60,70])

print(x.shape)

x = x.reshape(
    x.shape[0],
    x.shape[1]
    ,1
)


#평가 예측

model = load_model('C:/Users/ddong40/ai_2/_save/keras52/lstm2/save_6.78.94798.h5')

loss = model.evaluate(x, y)
x_test = np.array([50, 60, 70]).reshape(1, 3, 1)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('[50, 60, 70]의 결과 :', y_predict)

