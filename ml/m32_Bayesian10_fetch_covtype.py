from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time
import ssl
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

scaler = LabelEncoder()
y = scaler.fit_transform(y)

from sklearn.decomposition import PCA

print(x.shape) #(581012, 54)


# pca = PCA(n_components=54)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print( np.argmax(cumsum >=1.0)+1) #52
# print( np.argmax(cumsum >=0.999)+1) #5
# print( np.argmax(cumsum >=0.99)+1) #4
# print( np.argmax(cumsum >=0.95)+1) #2



print(datasets.DESCR)

# from tensorflow.keras.utils import to_categorical #케라스 (581012, 8)
# y = to_categorical(y) 

# y = pd.get_dummies(y) #(581012, 7)

# y = y.reshape(-1,1) #(581012, 7)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

print(x.shape)
print(y.shape) #(581012, 7)

# print(pd.value_counts(y,))

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth,num_leaves, min_child_samples,
              min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100 , 
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha' : reg_alpha,
    }
    
    model = XGBClassifier(**params, n_jobs=-1)
    
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)],
            #   eval_metric = 'logloss',
              verbose=0)
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333
)

n_iter= 300
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time, 2), '초')

# {'target': 0.8969389774790668, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 172.33110685027674, 'max_depth': 10.0, 'min_child_samples': 154.99926066277993, 'min_child_weight': 7.405953931588752, 'num_leaves': 25.95829923739064, 'reg_alpha': 4.245166069356461, 'reg_lambda': 1.9545683780729903, 'subsample': 0.5}}
# 300 번 걸린시간 : 2346.35 초