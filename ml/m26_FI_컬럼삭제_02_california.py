from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target
print(x)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    )


#2. 모델구성
model = RandomForestRegressor(random_state=Random_state)

model.fit(x_train, y_train)
print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('r2', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

percentiles = np.percentile(model.feature_importances_, 25)

indexs = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= percentiles:
        indexs.append(index)

x_train = np.delete(x_train, indexs, axis = 1)
x_test = np.delete(x_test, indexs, axis = 1)
model.fit(x_train, y_train)

print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('r2', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

# col = []
# for index, importance in enumerate(model.feature_importances_):
#     if importance < 0.01:
#         col.append(index)

# print(col)