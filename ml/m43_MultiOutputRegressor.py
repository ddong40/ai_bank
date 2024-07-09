#new gpu
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeClassifier

# 로지스틱리그레션은 분류
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, StackingClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

#1. 데이터

x, y = load_linnerud(return_X_y=True)

print(x.shape, y.shape) #(20, 3) (20, 3)
print(x)
print(y)

####### 데이터 형식 ########
#       x                   y
# [ 5. 162. 60.] -> [191. 36. 50.]
# ..............
# [2. 110. 43.] -> [138. 33. ] 

#2. 모델
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # RandomForestRegressor 스코어 : 3.5062
print(model.predict([[2, 110, 43]])) #[[155.82  34.35  63.26]]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # LinearRegression 스코어 :  7.4567
print(model.predict([[2, 110, 43]])) #[[187.33745435  37.08997099  55.40216714]]

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # LinearRegression 스코어 :  0.0008
print(model.predict([[2, 110, 43]])) #[[138.0005    33.002136  67.99897 ]]

# model = CatBoostRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 : ',
#       round(mean_absolute_error(y, y_pred), 4)) 
# print(model.predict([[2, 110, 43]])) 
# Currently only multi-regression, multilabel and survival objectives work with multidimensional target

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # MultiOutputRegressor 스코어 :  8.91
print(model.predict([[2, 110, 43]])) # [[178.6  35.4  56.1]]


model = MultiOutputRegressor(CatBoostRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2, 110, 43]])) # [[138.97756017  33.09066774  67.61547996]]


model = CatBoostRegressor(loss_function='MultiRMSE') #loss_function에 MutliRMSE사용하여 y 라벨이 다차원일 때 사용 가능하다.
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # MultiOutputRegressor 스코어 : 0.0638
print(model.predict([[2, 110, 43]])) # [[138.21649371  32.99740595  67.8741709 ]]