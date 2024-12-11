from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd

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
# acc 0.956140350877193
# 0.005446778200553036
# [0.06237093 0.01299331 0.03054104 0.02229684 0.00619588 0.00663481
#  0.03467701 0.0942553  0.00353745 0.00399233 0.01570823 0.00470082
#  0.00472369 0.04862523 0.0038361  0.00562107 0.00280699 0.00538868
#  0.00435311 0.00574178 0.12734821 0.01145023 0.1814384  0.11398163
#  0.01242737 0.01325251 0.03030373 0.11002503 0.01346049 0.00731179]

# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9473684210526315
# 0.011692723743333912
# [0.03490839 0.01682015 0.05309424 0.05222545 0.00785383 0.00688316
#  0.05853009 0.10753948 0.01875706 0.04127253 0.0041867  0.00646151
#  0.11371014 0.01352988 0.14337785 0.10586392 0.01108034 0.01628203
#  0.03079024 0.134048   0.01453233 0.0082527 ]