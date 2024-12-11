#가장 안좋은 컬럼들을 pca로 합친다. 
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#1 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, stratify=y
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
        
x_train1 = []     
for i in indexs : 
    x_train1.append(x_train[:,i])
x_train1 = np.array(x_train1).T

x_test1 = []     
for i in indexs : 
    x_test1.append(x_test[:,i])
    x_test1 = np.array(x_test1).T

print(x_train1.shape)
print(x_test1.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1)


x_train = np.delete(x_train, indexs, axis = 1)
x_test = np.delete(x_test, indexs, axis = 1)

x_train = np.concatenate([x_train, x_train1], axis=1)
x_test = np.concatenate([x_test, x_test1], axis=1)

model.fit(x_train, y_train)

print(indexs)


print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('acc', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)




# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9722222222222222
# 0.023994049111195584
# [0.14313907 0.03102579 0.01869279 0.02399405 0.031741   0.0576378
#  0.15946495 0.01065871 0.01751726 0.1640174  0.08038216 0.10883499
#  0.15289403]

# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9722222222222222
# 0.033801673687874595
# [0.16514941 0.02732637 0.02521907 0.04751502 0.1520093  0.16250766
#  0.08258577 0.12175253 0.18670432 0.02923056]