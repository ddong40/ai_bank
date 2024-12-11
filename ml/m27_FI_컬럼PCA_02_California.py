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
datasets = fetch_california_housing()
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



# ======================== RandomForestRegressor Random_state =  8888 ======================
# r2 0.8066038536752275
# 0.03823553020146351
# [0.52038394 0.05230074 0.04087402 0.02995461 0.03032005 0.13824539
#  0.09466481 0.09325643]

# ======================== RandomForestRegressor Random_state =  8888 ======================
# r2 0.8080235320408277
# 0.05074373356106208
# [0.52337069 0.05463583 0.04685164 0.1419798  0.1000519  0.09815077
#  0.03495937]
