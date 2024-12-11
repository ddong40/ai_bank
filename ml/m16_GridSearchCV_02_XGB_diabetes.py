import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import xgboost as xgb

print(x.shape) #442, 10


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

#kfold
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
start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()



print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.2861891833948846
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.3141311535904062
y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 
# accuracy_score :  0.3141311535904062
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.3141311535904062
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  149.31 초