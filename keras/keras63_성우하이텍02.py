# 19일 월요일 종가를 맞춰봐
# 제한시간 18일 일요일 23시 59분
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Concatenate, Bidirectional
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import time

path = 'C:/Users/ddong40/ai_2/_data/중간고사데이터/'

x_naver = pd.read_csv(path + 'NAVER 240816.csv', index_col=0,thousands=",")
x_hive = pd.read_csv(path + '하이브 240816.csv', index_col=0, thousands= ",")
y = pd.read_csv(path + '성우하이텍 240816.csv', index_col=0, thousands=",")

# print(x_naver.isnull().sum())
# print(x_hive.isnull().sum())
# print(y.isnull().sum())

# x_naver = x_naver.fillna(x_naver.mean())
# x_hive = x_hive.fillna(x_hive.mean())
# y = y.fillna(y.mean())

# print(x_naver.isnull().sum())
# print(x_hive.isnull().sum())
# print(y.isnull().sum())

print(x_hive)


x_naver = x_naver[:948]
x_hive = x_hive[:950]
y = y[:948]

x_naver = x_naver.sort_values(by=['일자'], ascending = True)
x_hive = x_hive.sort_values(by=['일자'], ascending = True)
y = y.sort_values(by=['일자'], ascending = True)

print(y)

x_naver = x_naver.drop(['전일비'], axis=1)
x_hive = x_hive.drop(['전일비'], axis=1)
x_naver = x_naver.drop(columns=x_naver.columns[4], axis=1)
x_hive = x_hive.drop(columns=x_hive.columns[4], axis=1)
y = y['종가']

x_naver = x_naver.astype(float)
x_hive = x_hive.astype(float)
y = y.astype(float)

print(y)
print(x_hive)
print(x_naver)


print(x_naver.shape) #(948, 14)
print(x_hive.shape) #(948, 14)
print(y.shape) #(948,)

x_naver_test = x_naver[-10:]
x_hive_test = x_hive[-10:]
y_test1 = y[10:]
x_naver = x_naver[:-10]
x_hive = x_hive[:-10]
print(x_naver_test.shape)
print(x_hive_test.shape)
print(y_test1)



size = 10

def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

xxx_naver = split_x(x_naver, size)

xxx_hive = split_x(x_hive, size)

xxx_naver_test = split_x(x_naver_test, size)
xxx_hive_test = split_x(x_hive_test, size)

print(xxx_naver.shape) #(939, 5, 14)
print(xxx_naver_test) #(939, 5, 14)
print(xxx_hive_test.shape) #(1, 10, 14)
print(xxx_naver_test.shape) #(1, 10, 14)


yyy =split_x(y_test1, size)

print(yyy.shape) 
print(yyy)

yyy= yyy[:, 0]

print(yyy)
print(yyy.shape) #(939,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(xxx_naver, xxx_hive, yyy, test_size=0.2, random_state=1024, shuffle=True)

print(x1_train.shape) #(743, 10, 14)
print(x1_test.shape) #(186, 10, 14)
print(x2_train.shape) #(743, 10, 14)
print(x2_test.shape) #(186, 10, 14)
print(y_train.shape) #(743,) 배누리씨 이러시면 안됩니다

x1_train = np.reshape(x1_train, (743, 10*14))
x1_test = np.reshape(x1_test, (186, 10*14))
x2_train = np.reshape(x1_train, (743, 10*14))
x2_test = np.reshape(x2_test, (186, 10*14))
xxx_naver_test = np.reshape(xxx_naver_test, (1, 10*14))
xxx_hive_test = np.reshape(xxx_hive_test, (1, 10*14))

StandardScaler()
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
xxx_naver_test = scaler.transform(xxx_naver_test)
xxx_hive_test = scaler.transform(xxx_hive_test)

x1_train = np.reshape(x1_train, (743, 10, 14))
x1_test = np.reshape(x1_test, (186, 10, 14))
x2_train = np.reshape(x1_train, (743, 10, 14))
x2_test = np.reshape(x2_test, (186, 10, 14))
xxx_naver_test = np.reshape(xxx_naver_test, (1, 10, 14))
xxx_hive_test = np.reshape(xxx_hive_test, (1, 10, 14))


# x_naver_test = np.reshape(x_naver_test, (10, 14))
# x_hive_test = np.reshape(x_hive_test, (10, 14))


#2-1 모델구성
input1 = Input(shape=(10, 14))
dense1 = Bidirectional(LSTM(10, activation='relu', name='bit1'))(input1)
dense2 = Dense(32, activation='relu', name='bit2')(dense1)
dense3 = Dense(128, activation='relu', name='bit3')(dense2)
dense4 = Dense(64, activation='relu', name='bit4')(dense3)
output1 = Dense(32, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)

# model1.summary()

#2-2 모델구성
input11 = Input(shape=(10, 14))
dense11 = Bidirectional(LSTM(10, activation='relu', name='bit11'))(input11)
dense21 = Dense(128, activation='relu', name='bit21')(dense11)
output11 = Dense(64, activation='relu', name='bit31')(dense21)
# model2 = Model(inputs=input11, outputs=output11)


merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(64, name='mg2')(merge1)
merge3 = Dense(32, name='mg3')(merge2)
output = Dense(10, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=output)

model.summary()


###########컴파일 훈련#########


model.compile(loss='mse', optimizer='adam')

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/중간고사가중치/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verboss =1,
    save_best_only=True,
    filepath=filepath
)

model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=16, verbose=1, validation_split=0.1, callbacks=[es, mcp])

end_time = time.time()

# model.save("./_save/_data/중간고사데이터/_전사영.h5")

loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([xxx_naver_test, xxx_hive_test])

print('로스 : ', loss)
print('시간 : ', round(end_time - start_time, 3), '초')
print('종가가격 : ', y_predict[0][0], '원')


#standard scaler
# 로스 :  1804872576.0
# 시간 :  406.9 초
# 종가가격 :  33457.06 원