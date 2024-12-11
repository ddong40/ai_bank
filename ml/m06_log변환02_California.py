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