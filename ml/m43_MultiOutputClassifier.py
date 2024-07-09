import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

np.random.seed(42)

# 다중분류 데이터 생성 함수
def create_multiclass_data_with_labels():
    # X 데이터 생성 (20, 3)
    X = np.random.rand(20, 3)
    
    # y 데이터 생성 (20, 3)
    y = np.random.randint(0, 5, size=(20, 3)) # 각 클래스 0부터 9까지 값
    
    # 데이터프레임으로 변환
    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])
    
    return X_df, y_df

X, y = create_multiclass_data_with_labels()
print("X 데이터 : ")
print(X)
print("\nY 데이터 : ")
print(y)
print(X.shape, y.shape)

from sklearn.multioutput import MultiOutputClassifier

#2. 모델

model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.284162,  0.253274,  0.901665]]))


model = MultiOutputClassifier(XGBClassifier())
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.195983,  0.045227,  0.325330]]))

model = MultiOutputClassifier(LGBMClassifier())
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.195983,  0.045227,  0.325330]]))

model = MultiOutputClassifier(CatBoostClassifier())
model.fit(X, y)
y_pred = model.predict(X)
y_pred = np.reshape(y_pred, (20, 3))
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.195983,  0.045227,  0.325330]]))

model = MultiOutputClassifier(CatBoostClassifier(loss_function='MultiClass'))
model.fit(X, y)
y_pred = model.predict(X)
y_pred = np.reshape(y_pred, (20, 3))
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.195983,  0.045227,  0.325330]]))