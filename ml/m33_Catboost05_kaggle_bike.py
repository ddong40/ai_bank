import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C:/Users/ddong40/ai_2/_data/kaggle/bike-sharing-demand/' #절대경로(경로가 풀로 다 들어간 경우)
# path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand' #위와 다 같음
# path = 'C://Users//ddong40//ai//_data//bike-sharing-demand' #위와 다 같음

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) #(6493, 1)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object'
    
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())

######### 결측치 확인 ###########

print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())


###### x와 y분리
x = train_csv.drop(['casual','registered','count'], axis=1) #이 리스트의 컬럼들을 axis 1에 넣어 드랍해주세요 라는 뜻
print(x)
y = train_csv['count']

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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