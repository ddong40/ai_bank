#가장 안좋은 컬럼들을 pca로 합친다. 
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')


#1 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

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

# {'target': np.float64(0.850821160981522), 'params': {'colsample_bytree': np.float64(0.5), 'learning_rate': np.float64(0.1), 'max_bin': np.float64(224.60789371016065), 'max_depth': np.float64(10.0), 
# 'min_child_samples': np.float64(130.6738357285239), 'min_child_weight': np.float64(19.905777296944695), 'num_leaves': np.float64(34.90669597189142), 'reg_alpha': np.float64(0.01), 'reg_lambda': np.float64(10.0), 'subsample': np.float64(1.0)}}
# 300 번 걸린시간 : 126.85 초

#############################
exit()















exit()
Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    )


#2. 모델구성
model = RandomForestRegressor(random_state=Random_state)

model.fit(x_train, y_train)
print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('r2', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

percentiles = np.percentile(model.feature_importances_, 25)

indexs = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= percentiles:
        indexs.append(index)
        
x_train1 = []     
for i in indexs : 
    x_train1.append(x_train[:,i])
x_train1 = np.array(x_train1).T

x_test1 = []     
for i in indexs : 
    x_test1.append(x_test[:,i])
x_test1 = np.array(x_test1).T

print(x_train1.shape)
print(x_test1.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1)


x_train = np.delete(x_train, indexs, axis = 1)
x_test = np.delete(x_test, indexs, axis = 1)

x_train = np.concatenate([x_train, x_train1], axis=1)
x_test = np.concatenate([x_test, x_test1], axis=1)

model.fit(x_train, y_train)

print(indexs)


print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('r2', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)



# ======================== RandomForestRegressor Random_state =  8888 ======================
# r2 0.8066038536752275
# 0.03823553020146351
# [0.52038394 0.05230074 0.04087402 0.02995461 0.03032005 0.13824539
#  0.09466481 0.09325643]

# ======================== RandomForestRegressor Random_state =  8888 ======================
# r2 0.8080235320408277
# 0.05074373356106208
# [0.52337069 0.05463583 0.04685164 0.1419798  0.1000519  0.09815077
#  0.03495937]
