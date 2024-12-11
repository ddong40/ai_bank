from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)


datasets = load_boston()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=337)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# learning_rate = 0.001 #디폴트

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience= 15,
    verbose=1,
    factor = 0.9   #factor는 learning rate * factor값
)

learning_rate = 0.01


model.compile(loss='mse', optimizer = Adam(learning_rate=learning_rate))

model.fit(x_train, y_train,
          validation_split=0.2, 
          epochs=1000,
          batch_size=32, callbacks = [es, rlr])

#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=0)

print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

# 0.001
# 로스 :34.11334991455078
# r2 : 0.6366904421717885

# 0.005
# 로스 :34.42169952392578
# r2 : 0.6334064828710586

# 0.009
# 로스 :34.517059326171875
# r2 : 0.6323908977031838