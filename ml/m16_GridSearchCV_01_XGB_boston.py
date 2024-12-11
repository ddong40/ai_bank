#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
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
     'min_samples_leaf' : [3, 10]}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10]}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10]}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10]}, #4
] #48



n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)



model = GridSearchCV(xgb.XGBRegressor(device = 'cuda:0'), parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, #24개의 코어가 한번에 돌아감cpu
                     ) 
start_time = time.time()\
    
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1, 'tree_method': 'gpu_hist'}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.8545922404692317
print('모델의 점수 : ', model.score(x_test, y_test)) 

y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 

print('걸린시간 : ', round(end_time - start_time, 2), '초') 
    
# 13   ★★★
# 로스 :  12.33582878112793
# r2 score : 0.8682481618376505


# 6
# 로스 :  18.08567237854004
# r2 score : 0.8068373919796679
    
# 3   
# 로스 :  59.77436447143555
# r2 score : 0.36158462792555557   
    
    
# 2    
# 로스 :  72.95055389404297
# r2 score : 0.22085739627870626



# parameters = [
#     {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
#      'min_samples_leaf' : [3, 10]}, #12
#     {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
#      'min_samples_leaf' : [3, 5, 7, 10]}, #16
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
#      'min_samples_leaf' : [2, 3, 5, 10]}, #16
#     {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10]}, #4
# ] #48

