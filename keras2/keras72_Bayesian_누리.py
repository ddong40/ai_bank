import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score

from bayes_opt import BayesianOptimization

import time

#1. 데이터
x, y = load_diabetes(return_X_y=True)

# y = pd.get_dummies(y)
# y = y.reshape(-1,1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
                                                    # stratify=y
                                                    )


print(x_train.shape, y_train.shape) # (353, 10) (353,)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lebal  = LabelEncoder()

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8):
    activation = lebal.inverse_transform([int(activation)])[0]
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(int(node1), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(int(node2), activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node3), activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(int(node4), activation=activation, name='hidden4')(x)
    x = Dense(int(node5), activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=le.inverse_transform([int(optimizer)])[0], metrics=['mae'], loss='mse')
    
    model.fit(x_train, y_train, epochs=100, 
            #   callbacks = [mcp, es, rlr],
              validation_split = 0.1,
            #   batch_size=batchs,
              verbose=0,
              )
    
    y_pre = model.predict(x_test)
    
    result = r2_score(y_test, y_pre)
    
    return result     


def create_hyperparameter():
    # batchs = (8, 64)
    optimizers = ['adam', 'rmsprop', 'adadelta']
    optimizers = (0, max(le.fit_transform(optimizers)))
    dropouts = (0.2, 0.5)
    activations = ['relu', 'elu', 'selu', 'linear']
    activations = (0, max(lebal.fit_transform(activations)))
    node1 = (16, 128)
    node2 = (16, 128)
    node3 = (16, 128)
    node4 = (16, 128)
    node5 = (16, 128)
    return {
        # 'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,      
        }


hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': ([100, 200, 300, 400, 500],), 'optimizer': (['adam', 'rmsprop', 'adadelta'],), 'drop': ([0.2, 0.3, 0.4, 0.5],), 'activation': (['relu', 'elu', 'selu', 'linear'],), 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1, 
                             )

bay = BayesianOptimization(
    f=build_model,
    pbounds=hyperparameters,
    random_state=333    
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

# {'target': 0.2936119129356415, 'params': {'activation': 3.0, 'drop': 0.2, 'node1': 128.0, 'node2': 44.285812080627885, 'node3': 74.32881255322306, 'node4': 96.0063162967301, 'node5': 128.0, 'optimizer': 1.9999879296867078}}
# 100 번 걸린 시간 : 1188.4 초