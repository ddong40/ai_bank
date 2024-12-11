from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    stratify=y
                                                    )


#2. 모델

early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True # error
    # AttributeError: `best_iteration` is only defined when early stopping is used.
)
model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha=0, #L1규제
    reg_lambda = 1, #L2규제
    eval_metrics='logloss', #다중분류에는 mlogloss, 이진분류에는 logloss와 error를 사용하면 된다. 
    callbacks = [early_stop],
    random_state = 3377
)



#3 훈련 
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],
        #   eval_metrics = 'mlogloss',
          verbose=True)

results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print('acc :', acc)

print(model.feature_importances_)

# [3.1095673e-03 2.6227774e-02 3.9051243e-03 1.6615657e-05 2.0730145e-02
#  8.1776241e-03 5.2664266e-03 8.4118381e-02 7.8737978e-03 4.9598357e-03
#  1.3275798e-02 1.4882818e-03 5.9160744e-03 6.1758894e-02 1.0046216e-02
#  7.9790084e-03 9.9304272e-03 1.4152890e-02 1.5800500e-02 5.6784889e-03
#  5.2224431e-02 4.9101923e-02 7.0977546e-02 2.1681710e-01 2.2077497e-02
#  1.6301911e-02 2.9043932e-02 1.9351867e-01 1.4708905e-02 2.4816213e-02]

thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds)

# [1.6615657e-05 1.4882818e-03 3.1095673e-03 3.9051243e-03 4.9598357e-03
#  5.2664266e-03 5.6784889e-03 5.9160744e-03 7.8737978e-03 7.9790084e-03
#  8.1776241e-03 9.9304272e-03 1.0046216e-02 1.3275798e-02 1.4152890e-02
#  1.4708905e-02 1.5800500e-02 1.6301911e-02 2.0730145e-02 2.2077497e-02
#  2.4816213e-02 2.6227774e-02 2.9043932e-02 4.9101923e-02 5.2224431e-02
#  6.1758894e-02 7.0977546e-02 8.4118381e-02 1.9351867e-01 2.1681710e-01]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBClassifier(n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha=0, #L1규제
    reg_lambda = 1, #L2규제
    eval_metrics='logloss', #다중분류에는 mlogloss, 이진분류에는 logloss와 error를 사용하면 된다. 
    # callbacks = [early_stop],
    random_state = 3377)
    
    select_model.fit(select_x_train, y_train, eval_set=[(select_x_test,y_test)], verbose=False)
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, acc: %.2f%%' %(i,select_x_train.shape[1], score*100) )
    
# Trech=0.002, n=30, acc: 97.37%
# Trech=0.003, n=29, acc: 96.49%
# Trech=0.005, n=28, acc: 97.37%
# Trech=0.005, n=27, acc: 97.37%
# Trech=0.006, n=26, acc: 96.49%
# Trech=0.007, n=25, acc: 96.49%
# Trech=0.008, n=24, acc: 96.49%
# Trech=0.008, n=23, acc: 96.49%
# Trech=0.010, n=22, acc: 96.49%
# Trech=0.010, n=21, acc: 96.49%
# Trech=0.010, n=20, acc: 95.61%
# Trech=0.012, n=19, acc: 96.49%
# Trech=0.012, n=18, acc: 96.49%
# Trech=0.014, n=17, acc: 96.49%
# Trech=0.015, n=16, acc: 98.25%
# Trech=0.018, n=15, acc: 96.49%
# Trech=0.018, n=14, acc: 98.25%
# Trech=0.019, n=13, acc: 97.37%
# Trech=0.019, n=12, acc: 96.49%
# Trech=0.022, n=11, acc: 97.37%
# Trech=0.027, n=10, acc: 98.25%
# Trech=0.028, n=9, acc: 98.25%
# Trech=0.031, n=8, acc: 97.37%
# Trech=0.032, n=7, acc: 97.37% ★
# Trech=0.039, n=6, acc: 93.86%
# Trech=0.051, n=5, acc: 93.86%
# Trech=0.085, n=4, acc: 94.74%
# Trech=0.089, n=3, acc: 93.86%
# Trech=0.091, n=2, acc: 93.86%
# Trech=0.303, n=1, acc: 85.96%