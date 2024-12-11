import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
import time

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3434, shuffle=True)

parameters = {
    'learning_rate' : (0.01, 0.2),
    'depth' : (4, 12), # 6,
    'l2_leaf_reg' : (1, 10),
    'bagging_temperature' : (0, 5),
    'border_count' : (32, 255), 
    'random_strength': (1, 10),
    # 'n_jobs' : -1,  # CPU
}

def cat_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature,
              border_count, random_strength):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        'bagging_temperature' : bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : int(round(random_strength))
    }
    
    model = CatBoostRegressor(**params, task_type='GPU',
                              devices = '0', early_stopping_rounds=100)
    
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)]
              ,verbose=0)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=cat_hamsu,
    pbounds=parameters,
    random_state=333
)

n_iter = 300
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time,2 ), '초')