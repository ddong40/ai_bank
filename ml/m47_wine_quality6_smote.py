# 실습 : smote 써서 비교하기 
# y class 7개 그대로 

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

path = "C:/Users/ddong40/ai_2/_data/kaggle/wine_quality/"

# 만들어보기 : y는 quality

train = pd.read_csv(path + 'train.csv', index_col=0)
# print(train.head())
# print(train.info())
# print(train.isna().sum())

x = train.drop(['quality'], axis=1)
y = train['quality']

le = LabelEncoder()
x['type'] = le.fit_transform(x['type'])
print(le.transform(['red', 'white']))       # [0, 1]

# le2 = LabelEncoder()
# y = le2.fit_transform(y)
# y = y - 3
print(y.value_counts().sort_index())
# quality
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5

x = train.drop(['quality'], axis=1)
y = train['quality']

print(y.value_counts().sort_index())

le = LabelEncoder()
x['type'] = le.fit_transform(x['type'])

y = y - 3


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9978, test_size=0.2)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=7777, k_neighbors=1)

x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=4525,
)

model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose=0)

y_pre = model.predict(x_test)
acc = accuracy_score(y_pre, y_test)

print('acc :', acc)     
print('F1 :',f1_score(y_pre, y_test, average='macro'))     

data = y.groupby(train['quality']).count()
# print(data)

plt.bar(data.index, data.values)
plt.show()

"""
기본
acc : 0.6827272727272727

5개로 줄이기
acc : 0.6846435100548446

4개로 줄이기
acc : 0.678030303030303

3개로 줄이기
acc : 0.7076023391812866

3개로 합치기
acc : 0.7127272727272728
F1 : 0.7040043067018052

smote 
acc : 0.6563636363636364
F1 : 0.3921092622161506

"""

