import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time


path = './_data/kaggle/jena/'



train_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
train_csv2 = pd.read_csv(path + 'jena_climate_2009_2016.csv')
# sampleSubmission = pd.read_csv(path + 'sample_submission_jena.csv', index_col=0)


x = train_csv
sampleSubmission = train_csv2[-144:]
sampleSubmission = sampleSubmission[['Date Time', 'T (degC)']]

print(x.shape) #(420551, 14)
y2 = x[-144:]
y2 = y2['T (degC)']

print(y2.shape)


x = x[:-144]
y = x['T (degC)']
x = x.drop(['T (degC)'], axis=1)

 
print(x.shape)  #(420407, 13)
print(y.shape) #(420407,)

size = 144
size2 = 144

def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

def split_y(dataset, size2):
    aaa= []
    for i in range(len(dataset) - size2 + 1):
        subset = dataset[i : (i + size2)]
        aaa.append(subset)
    return np.array (aaa)

xxx = split_x(x, size)

print(xxx.shape) #(420264, 144, 13)

yyy = split_y(y, size2)
print(yyy)
print(yyy.shape) #(420264, 144)

yyy = yyy[1:]
print(yyy.shape) #(420263, 144)
print(yyy) 

x_test2 = xxx[-1]
print(x_test2)
print(x_test2.shape) #(144, 13)

xxx = xxx[:-1]
print(xxx.shape) #(420263, 144, 13)

x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, test_size=0.2, shuffle=True, random_state=100)

print(x_train.shape) #(315197, 144, 13)
print(x_test.shape) #(105066, 144, 13)

## 데이터 전처리 ### 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
x_test2 = np.reshape(x_test2, (1, x_test2.shape[0]*x_test2.shape[1]))

print(x_train.shape)  #(315197, 1872)
print(x_test.shape) #(105066, 1872)
print(x_test2.shape) #(1, 1872)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test2 = scaler.transform(x_test2)

x_train = np.reshape(x_train, (x_train.shape[0],144 ,13))
x_test = np.reshape(x_test, (x_test.shape[0],144 ,13))
x_test2 = np.reshape(x_test2, (1,144 ,13))

print(x_train.shape)  #(315197, 144, 13)
print(x_test.shape) #(105066, 144, 13)
print(x_test2.shape) #(1, 144, 13)


model = Sequential()
model.add(Bidirectional(LSTM(32,), input_shape=(144, 13)))
# model.add(LSTM(64, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(64, activation='relu'))
model.add(Dense(144))

#컴파일 훈련
model.compile(loss= 'mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=300, verbose=1, validation_split = 0.1, callbacks=[es] ) 
model.save("./_data/kaggle/jena/keras55_jena_save_model2.h5")


loss = model.evaluate(x_test, y_test, batch_size=300)
# x_test2 = x_test2.reshape(1, 144, 13)
y_predict = model.predict(x_test2, batch_size=300)

print(y_predict.shape, y_test.shape)

y_predict = np.reshape(y_predict, (144,1))

def RMSE(y2, y_predict):
    return np.sqrt(mean_squared_error(y2, y_predict))
rmse = RMSE(y2, y_predict)


# print(y_predict.shape)

sampleSubmission['T (degC)'] = y_predict

sampleSubmission.to_csv(path+'jena_전사영_submission.csv') 


print('로스 : ', loss)
print('y_predict', y_predict)
print('RMSE :', rmse)


# RMSE : 1.364764831074311