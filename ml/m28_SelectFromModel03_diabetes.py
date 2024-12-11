#가장 안좋은 컬럼들을 pca로 합친다. 
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score


#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    )


#2. 모델구성
#2. 모델

early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    # metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True # error
    # AttributeError: `best_iteration` is only defined when early stopping is used.
)


model = XGBRegressor(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha=0, #L1규제
    reg_lambda = 1, #L2규제
    # eval_metrics='logloss', #다중분류에는 mlogloss, 이진분류에는 logloss와 error를 사용하면 된다. 
    # callbacks = [early_stop],
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

acc = r2_score(y_test, y_pred)

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
    
    select_model = XGBRegressor(n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha=0, #L1규제
    reg_lambda = 1, #L2규제
    # eval_metrics='logloss', #다중분류에는 mlogloss, 이진분류에는 logloss와 error를 사용하면 된다. 
    # callbacks = [early_stop],
    random_state = 3377)
    
    select_model.fit(select_x_train, y_train, eval_set=[(select_x_test,y_test)], verbose=False)
    
    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, acc: %.2f%%' %(i,select_x_train.shape[1], score*100) )

# Trech=0.041, n=10, acc: 38.36% ★
# Trech=0.070, n=9, acc: 27.66%
# Trech=0.071, n=8, acc: 27.74%
# Trech=0.073, n=7, acc: 33.66%
# Trech=0.076, n=6, acc: 31.91%
# Trech=0.083, n=5, acc: 23.55%
# Trech=0.084, n=4, acc: 7.88%
# Trech=0.112, n=3, acc: 14.16%
# Trech=0.148, n=2, acc: -14.82%
# Trech=0.243, n=1, acc: -28.94%