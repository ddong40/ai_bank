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
print('r2', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)


# pca
# ======================== RandomForestRegressor Random_state =  8888 ======================
# r2 0.5249893823292553
# 0.06370802782599651
# [0.06323818 0.25060367 0.07964231 0.06603324 0.06270492 0.33727946
#  0.07663358 0.06386464]
