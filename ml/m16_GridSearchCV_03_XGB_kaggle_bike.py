# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score
import time 
from sklearn.preprocessing import StandardScaler

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

print(x.shape) #(10886, 8)
print(test_csv.shape) #(6493, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10]}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10]}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10]}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10]}, #4
] #48

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import xgboost as xgb

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)


model = GridSearchCV(xgb.XGBRegressor(device = 'cuda:0'), parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, #24개의 코어가 한번에 돌아감cpu
                     ) 
start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()



print('최적의 매개변수 : ', model.best_estimator_) 
# 최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device='cuda:0', early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=100,
#              n_jobs=-1, num_parallel_tree=None, ...)

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.2996530550577218
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.34094937685113036
y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 
# accuracy_score :  0.34094937685113036
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.34094937685113036
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  281.63 초