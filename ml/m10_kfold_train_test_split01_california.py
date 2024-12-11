from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

#1 데이터

datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot
# df.plot.box()
# plt.show()

# print(df.info())
# print(df.describe())

# df['Population'].plot.box() #시리즈에서 이거 됨
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb


#kfold 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

model = xgb.XGBRFRegressor() 

scores  = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

# kfold
# ACC :  [0.69710258 0.69613789 0.68712194 0.71058927 0.71272569] 
# 평균 ACC :  0.7007

y_predict = cross_val_predict(model, x_test, y_test)

acc = r2_score(y_test, y_predict)
print('cross_val_predict r2 : ', acc)

# cross_val_predict r2 :  0.6722020540460287

exit()
################################# Pupulation 로그변환 #########################################

x['Population'] = np.log1p(x['Population']) #지수변환 np.exp1m  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

################################# y로그변환 #########################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#####################################################################################

# model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)

model = LinearRegression()

#3 훈련
model.fit(x_train, y_train)

#4 평가 예측
score = model.score(x_test, y_test) #r2스코어 

print('score : ', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2 :', r2)

# 로그 변환 전 
# score :  0.6495152533878351
# r2 : 0.6495152533878351

# 로그변환 후 
# score :  0.6584197269397019
# r2 : 0.6584197269397019

# x의 pop 변환 후 
# score :  0.6495031475648194
# r2 : 0.6495031475648194

# x, y 둘다 변환
# score :  0.6584197269397019
# r2 : 0.6584197269397019



### linear ####
# score :  0.606572212210644
# r2 : 0.606572212210644

#y만
# score :  0.6295290651919585
# r2 : 0.6295290651919585

#x만 
# score :  0.606598836886877
# r2 : 0.606598836886877

# x y 둘다
# score :  0.6294707351612604
# r2 : 0.6294707351612604