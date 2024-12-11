#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston, load_iris, load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

import time
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import warnings 
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_digits()

x = dataset.data
y = dataset.target
 
print(x.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=3333, stratify=y)

print(x_train.shape, y_train.shape) #(1437, 64) (1437,)
print(x_test.shape, y_test.shape)  #(360, 64) (360,)



parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], #0.2, 0.3], 
    'max_depth' : [3, 4, 5,6,8]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0]}
] #25*4*cv

# parameters = [
#     {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
#      'min_samples_leaf' : [3, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #36
#     {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
#      'min_samples_leaf' : [3, 5, 7, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
#      'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
#     {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #12
# ] #134

### 경우의 수를 100개로 늘려서 랜덤서치 ###
# 러닝레이트 반드시 넣고
# 다른 파라미터도 두개 더 넣기

early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name = 'mlogloss',
    data_name = 'validation_0',
    save_best = True # error
)




n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)



model = HalvingRandomSearchCV(xgb.XGBClassifier( tree_method = 'hist',
    device = 'cuda:0',
    n_estimators=50,
    eval_metric = 'mlogloss',
    callbacks = [early_stop]), 
                    parameters, cv=kfold,
                    verbose=1,
                    refit=True,
                    #  n_iter = 9,
                    random_state= 333,
                    factor=3,
                    min_resources=30,
                    max_resources=1437,
                    aggressive_elimination=True,
                     )

start_time = time.time()
    
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = False,
          )

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 10, 'max_depth': 12, 'learning_rate': 0.01}
# 최적의 파라미터 :  {'subsample': 0.9, 'learning_rate': 0.2}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.8023263024659932
# 최고의 점수 :  0.9380262249827467
# early stopping
# 최고의 점수 :  0.9429798328349055
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.8550253794199509
# 모델의 점수 :  0.9694444444444444
# early stopping
# 모델의 점수 :  0.9666666666666667

y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 
# accuracy_score :  0.8550253794199509
# accuracy_score :  0.928909685209077
# early stopping
# accuracy_score :  0.9485441531037129
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.8550253794199509
# 최적 튠 ACC:  0.928909685209077
# early stopping
# 최적 튠 ACC:  0.9485441531037129
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  76.34 초
# 걸린시간 :  179.83 초
# early stopping
# 걸린시간 :  171.68 초


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

