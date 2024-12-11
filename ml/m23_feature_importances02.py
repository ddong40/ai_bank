### 회귀 ###
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes

#1 데이터

x, y = load_diabetes(return_X_y=True)

print(x.shape, y.shape)
# (442, 10) (442,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123)



#2 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

print('random_state : ', random_state)

for model in models:
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('r2', model.score(x_test, y_test)) 
    print(model.feature_importances_)
    
# =================== DecisionTreeRegressor ====================
# acc -0.02071233161542274
# [0.06216979 0.01445445 0.28596272 0.07156056 0.02075522 0.05880889
#  0.03121231 0.02276049 0.34071023 0.09160533]
# =================== RandomForestRegressor ====================
# acc 0.4397566348634161
# [0.05771616 0.01352985 0.30321945 0.09357224 0.04353204 0.04987969
#  0.05018382 0.02623353 0.28717917 0.07495404]
# =================== GradientBoostingRegressor ====================
# acc 0.4282356823245185
# [0.06472103 0.01560296 0.33332919 0.07799523 0.02486526 0.0558619
#  0.03286388 0.02978304 0.31735323 0.04762427]
# =================== XGBRegressor ====================
# acc 0.36474843193516526
# [0.03583801 0.0660802  0.19658518 0.05924349 0.04499551 0.04450775
#  0.08593133 0.06912719 0.31810254 0.07958876]