
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import tensorflow as tf
import random as rn 
from tensorflow.keras.optimizers import Adam
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape, x_test.shape)

##원핫 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
def build_model(drop=0.5, optimizer=Adam(learning_rate=0.01), activation='relu', node1=128, node2=64, node3=32, node4 = 16, node5=8,lr=0.001):
    inputs = Input(shape=(28*28), name='inputs')
    x = Dense(node1, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name= 'hidden2')(x)
    x = Dropout(drop)(x) 
    x = Dense(node3, activation=activation, name= 'hidden3')(x)
    x = Dropout(drop)(x) 
    x = Dense(node4, activation=activation, name= 'hidden4')(x) 
    x = Dense(node5, activation=activation, name= 'hidden5')(x) 
    outputs = Dense(10, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'], loss = 'mse')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5}
hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear'], 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16],
#  'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=5
                           ,n_iter = 10, verbose=1)
import time
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

filepath = 'C:/Users/ddong40/ai_2/_save/keras71'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only=True,
    filepath=filepath    
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience=15,
    verbose=1,
    factor=0.9
)


model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split = 0.3, callbacks=[es, mcp, rlr])
end = time.time()
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))
