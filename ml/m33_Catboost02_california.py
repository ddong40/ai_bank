import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(y)
print(np.unique(y))

print(x.shape, y.shape)
# (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=3434)

parmeters = {
    'learning_rate' : (0.01, 0.2),
    'depth' : (4, 12), # 6,
    'l2_leaf_reg' : (1, 10),
    'bagging_temperature' : (0, 5),
    'border_count' : (32, 255), 
    'random_strength': (1, 10),
    # 'n_jobs' : -1,  # CPU
}

# cat_features = list(range(x_train.shape[1]))
# print(cat_features) #[0, 1, 2, 3, 4, 5, 6, 7]

def xgb_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature,
              border_count, random_strength):
    params = {
        'n_estimators' : 100 , 
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        'bagging_temperature' : bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : int(round(random_strength))
    }
    
    model = CatBoostRegressor(**params, task_type='GPU', devices='0',early_stopping_rounds=100,) #cat_features= cat_features)
    
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)],
            #   eval_metric = 'logloss',
              verbose=0)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=parmeters,
    random_state=333
)

n_iter= 300
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time, 2), '초')

# {'target': 0.8438523332020054, 'params': {'bagging_temperature': 0.0, 'border_count': 164.25483198392402, 'depth': 10.219469104849914, 'l2_leaf_reg': 1.326012672314205, 'learning_rate': 0.2, 'random_strength': 1.0}}
# 300 번 걸린시간 : 369.96 초