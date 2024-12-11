from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_wine()
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


#2. 모델구성
model = RandomForestClassifier(random_state=Random_state)

model.fit(x_train, y_train)
print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('acc', model.score(x_test, y_test))
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
print('acc', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

# col = []
# for index, importance in enumerate(model.feature_importances_):
#     if importance < 0.01:
#         col.append(index)

# print(col)

# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9722222222222222
# 0.023994049111195584
# [0.14313907 0.03102579 0.01869279 0.02399405 0.031741   0.0576378
#  0.15946495 0.01065871 0.01751726 0.1640174  0.08038216 0.10883499
#  0.15289403]
# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9722222222222222
# 0.05536880331325454
# [0.15658529 0.02522465 0.03023223 0.0553688  0.18124215 0.19567425
#  0.07195151 0.13196233 0.1517588 ]