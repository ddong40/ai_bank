import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
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
model.add(SimpleRNN(128, activation= 'relu', input_shape=(3, 1)))  #행 무시 열 우선 즉 7을 빼준다. #Rnn은 Dense와 바로 연결이 가능하다.  
# model.add(Dropout(0.25))
# model.add(LSTM(128, activation= 'relu'))
# model.add(LSTM(128, activation= 'relu', input_shape=(3,1) ))
# model.add(Dropout(0.25))
# model.add(SimpleRNN(256, activation= 'relu'))
# model.add(Dropout(0.25))
# model.add(SimpleRNN(128, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=80,
    restore_best_weights=True
)


model.fit(x, y, epochs=500, batch_size=8, verbose=1, validation_split=0.25, callbacks=[es])

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