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
datasets = load_breast_cancer()
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
# acc 0.9473684210526315
# 0.011335618616166724
# [0.06091001 0.01222387 0.04618342 0.04679902 0.00639301 0.0078151
#  0.06161106 0.11438608 0.01548929 0.05291985 0.00585098 0.00727029
#  0.05418605 0.01075713 0.13825994 0.11765469 0.01479515 0.01233743
#  0.04305972 0.12549465 0.0119141  0.00962625 0.02406289]