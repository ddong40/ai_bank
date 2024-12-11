import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Bidirectional, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) #(7, 3, 1)

#2. 모델구성
# A 3D tensor, with shape [batch, timesteps, feature].
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
# model.add(Conv1D(filters=10, kernel_size=2))
model.add(Flatten())
# model.add(Dense(7))
model.add(Dense(1))

model.summary()

#3. 컴파일 훈련
'''
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     verbose=1,
#     patience=80,
#     restore_best_weights=True
# )


model.fit(x, y, epochs=2000, batch_size=55, verbose=1, validation_split=0.25)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results[0])
print('accuracy : ', results[1])


x_pred = np.array([8, 9, 10]).reshape(1, 3, 1) #백터 (3, ) -> (1,3,1)로 reshape
y_pred = model.predict(x_pred)
# acc = accuracy_score(y, y_pred)

print('[8,9,10]의 결과 : ', y_pred)
# [8,9,10]의 결과 :  [[0.35999376]]

#GRU
# loss :  0.0136331832036376
# accuracy :  0.0
# [8,9,10]의 결과 :  [[10.966791]]

#LSTM
# loss :  0.0015269698342308402
# accuracy :  0.0
# [8,9,10]의 결과 :  [[10.9709635]]

# SimpleRNN
# loss :  0.03974154591560364
# accuracy :  0.0
# [8,9,10]의 결과 :  [[11.139857]]
'''