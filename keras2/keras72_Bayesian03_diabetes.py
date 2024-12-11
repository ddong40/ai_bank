import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from bayes_opt import BayesianOptimization
import time
from sklearn.preprocessing import LabelEncoder

#1. 데이터 
x, y = load_diabetes(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x, y ,shuffle=True, random_state=336, test_size=0.2)

print(x_train.shape, y_train.shape) #(353, 10) (353,)

le = LabelEncoder()
label = LabelEncoder()
optimizer = ['adam', 'rmsprop', 'adadelta']
activation = ['relu', 'elu', 'selu', 'linear']
optimizer = le.fit_transform(optimizer)
activation = label.fit_transform(activation)

bayesian_params = {
    'batch_size' : (100, 500),
    'optimizer' : (optimizer[0], optimizer[1], optimizer[2]),
    'drop' : (0.2, 0.5),
    'activation' : (activation[0], activation[1], activation[2], activation[3]),
    'node1' : (16, 128),
    'node2' : (16, 128),
    'node3' : (16, 128),
    'node4' : (16, 128),
    'node5' : (16, 128)}

# def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, node4 = 16, node5=8,lr=0.001):
#     inputs = Input(shape=(10,), name='inputs')
#     x = Dense(node1, activation=activation, name= 'hidden1')(inputs)
#     x = Dropout(drop)(x)
#     x = Dense(node2, activation=activation, name= 'hidden2')(x)
#     x = Dropout(drop)(x) 
#     x = Dense(node3, activation=activation, name= 'hidden3')(x)
#     x = Dropout(drop)(x) 
#     x = Dense(node4, activation=activation, name= 'hidden4')(x) 
#     x = Dense(node5, activation=activation, name= 'hidden5')(x) 
#     outputs = Dense(1, activation='linear', name='outputs')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
    
#     model.compile(optimizer=optimizer, metrics=['mae'], loss = 'mse')
    
#     return model

def dnn_hamsu(batch_size, optimiezr, drop, activation, node1, node2, node3, node4, node5):
    params = { 
        'batch_size' : batch_size,
        'optimizer' : (optimizer[0], optimizer[1], optimizer[2]),
        'drop' : drop,
        'activation' : (activation[0], activation[1], activation[2], activation[3]),
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5}
    model = build_model()

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, node4 = 16, node5=8,lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
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
model.fit(x_train, y_train, epochs=1000)
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))