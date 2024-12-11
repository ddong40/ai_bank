import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# x = np.concatenate([x_train, x_test], axis=0)

# print(x.shape) #(70000, 28, 28, 1)

x_train = np.reshape(x_train, (60000, 28*28))
x_test = np.reshape(x_test, (10000, 28*28))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

from sklearn.decomposition import PCA

# pca = PCA(n_components=784)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print( np.argmax(cumsum >= 1.0)+1) #784
# print( np.argmax(cumsum >= 0.99)+1) #459
# print( np.argmax(cumsum >= 0.999)+1) #674
# print( np.argmax(cumsum >= 0.95)+1) #188



list_a = np.array([188, 459, 674, 784])

for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    #2 모델 구성

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape = (i,))) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu')) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=16, input_shape=(32,), activation='relu')) 
    model.add(Dense(10, activation='softmax'))

    #3 컴파일 훈련

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')

    path1 = './_save/keras35/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path1, 'k30_', date, '_', filename])


    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only=True,
        filepath=filepath
        )


    model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.25, callbacks=[es, mcp])

    # 예측, 평가
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
    # y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

    # acc = accuracy_score(y_test, y_predict)

    print('로스 : ', loss[0])
    print('acc 스코어 :', loss[1])
    print('==============================================')
    