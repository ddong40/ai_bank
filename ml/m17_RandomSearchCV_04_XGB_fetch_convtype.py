from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

scaler = LabelEncoder()
y = scaler.fit_transform(y)

from sklearn.decomposition import PCA

print(x.shape) #(581012, 54)


# pca = PCA(n_components=54)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print( np.argmax(cumsum >=1.0)+1) #52
# print( np.argmax(cumsum >=0.999)+1) #5
# print( np.argmax(cumsum >=0.99)+1) #4
# print( np.argmax(cumsum >=0.95)+1) #2



print(datasets.DESCR)

# from tensorflow.keras.utils import to_categorical #케라스 (581012, 8)
# y = to_categorical(y) 

# y = pd.get_dummies(y) #(581012, 7)

# y = y.reshape(-1,1) #(581012, 7)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

print(x.shape)
print(y.shape) #(581012, 7)

# print(pd.value_counts(y,))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, random_state=500,
                                                    stratify=y)
parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #36
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #12
] #134


from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import xgboost as xgb

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)


model = RandomizedSearchCV(xgb.XGBClassifier(device = 'cuda:0'), parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, 
                     n_iter= 8
                     ) 
start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()



print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1}
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 12, 'learning_rate': 0.001}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.96761804510577
# 최고의 점수 :  0.8581660502909066
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.9714521755094547
# 모델의 점수 :  0.8611850559941252
y_predict = model.predict(x_test)

print('accuracy_score : ', accuracy_score(y_test, y_predict)) 
# accuracy_score :  0.9714521755094547
# accuracy_score :  0.8611850559941252
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', accuracy_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.9714521755094547
# 최적 튠 ACC:  0.8611850559941252
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  2278.34 초
# 걸린시간 :  477.08 초