#가장 안좋은 컬럼들을 pca로 합친다. 
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, accuracy_score
from bayes_opt import BayesianOptimization
import time
import warnings
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8,
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

# {'target': 0.5432960117030909, 'params': {'colsample_bytree': 0.7428034637694538, 'learning_rate': 0.1, 'max_bin': 100.57200029302341, 'max_depth': 8.715950802532177, 'min_child_samples': 121.73904618193866, 'min_child_weight': 49.91065307075642, 'num_leaves': 32.58073134660018, 'reg_alpha': 33.37500569022738, 'reg_lambda': 7.838044802621286, 'subsample': 0.7328645899429836}}
# 300 번 걸린시간 : 83.94 초