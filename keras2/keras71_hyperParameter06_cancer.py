import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings('ignore')


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x, y ,shuffle=True, random_state=336, test_size=0.2)

print(x_train.shape, y_train.shape) #(455, 30) (455,)


#2. 모델
def build_model(drop=0.5, optimizer=Adam(learning_rate=0.01), activation='relu', node1=128, node2=64, node3=32, node4 = 16, node5=8,lr=0.001):
    inputs = Input(shape=(30,), name='inputs')
    x = Dense(node1, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name= 'hidden2')(x)
    x = Dropout(drop)(x) 
    x = Dense(node3, activation=activation, name= 'hidden3')(x)
    x = Dropout(drop)(x) 
    x = Dense(node4, activation=activation, name= 'hidden4')(x) 
    x = Dense(node5, activation=activation, name= 'hidden5')(x) 
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
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

# 걸린시간 :  19.12
# model.best_params_ {'optimizer': 'rmsprop', 'node5': 32, 'node4': 32, 'node3': 16, 'node2': 128, 'node1': 64, 'drop': 0.2, 'batch_size': 400, 'activation': 'selu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000021975CC1CA0>
# model.best_score :  -28698.737109375
# 1/1 [==============================] - 0s 50ms/step - loss: 28443.4102 - mae: 149.7358
# model.score :  -28443.41015625
