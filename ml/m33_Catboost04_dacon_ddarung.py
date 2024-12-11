import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
import time

#1. 데이터
path = 'C:/Users/ddong40/ai_2/_data/dacon/따릉이/'       # 경로지정 #상대경로 

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #predict할 x데이터
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #predict 할 y데이터
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info() #train_csv의 정보를 알려주는 함수

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum()) 

train_csv = train_csv.dropna() #dropna는 결측치의 행을 삭제해라

test_csv = test_csv.fillna(test_csv.mean())  #fillna 채워라 #mean함수는 뭔디    #컬럼끼리만 평균을 낸다

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다. #axis는 축이다 
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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