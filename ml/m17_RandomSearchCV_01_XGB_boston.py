#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import time
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import warnings 
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #36
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #12
] #134

### 경우의 수를 100개로 늘려서 랜덤서치 ###
# 러닝레이트 반드시 넣고
# 다른 파라미터도 두개 더 넣기

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)



model = RandomizedSearchCV(xgb.XGBRegressor(device = 'cuda:0'), parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1,
                     n_iter = 9,
                     random_state= 3333
                     ) 
start_time = time.time()\
    
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 10, 'max_depth': 12, 'learning_rate': 0.01}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.8023263024659932
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.8550253794199509
y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 
# accuracy_score :  0.8550253794199509
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.8550253794199509
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  76.34 초

# 13   ★★★
# 로스 :  12.33582878112793
# r2 score : 0.8682481618376505





# parameters = [
#     {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
#      'min_samples_leaf' : [3, 10]}, #12
#     {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
#      'min_samples_leaf' : [3, 5, 7, 10]}, #16
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
#      'min_samples_leaf' : [2, 3, 5, 10]}, #16
#     {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10]}, #4
# ] #48

