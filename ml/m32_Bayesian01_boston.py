from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

x = df.drop(['target'], axis=1).copy()
y = df['target']

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8
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
    
    model = XGBRegressor(**params, n_jobs=-1)
    
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)],
            #   eval_metric = 'logloss',
              verbose=0)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
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

# {'target': 0.914225, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 120.90037228949565, 'max_depth': 10.0, 'min_child_samples': 45.93903949249824, 'min_child_weight': 8.705442991393975, 'num_leaves': 40.0, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 0.5}}
# 300 번 걸린시간 : 1119.33 초

#############################
exit()
datasets = load_boston()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot
# df.plot.box()
# plt.show()

# print(df.info()) #결측치 없고~
# print(df.describe())

# df['B'].plot.box() #시리즈에서 이거 됨
# plt.show()

# df['B'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

x['B'] = np.log1p(x['B']) #지수변환 np.exp1m  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

y_train = np.log1p(y_train)
y_test = np.log1p(y_train)

model = LinearRegression()