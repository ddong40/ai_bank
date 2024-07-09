#new gpu
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 로지스틱리그레션은 분류
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import PolynomialFeatures

#1. 데이터
x, y = load_breast_cancer(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=4444, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pf = PolynomialFeatures(degree=2, include_bias=True) #degree를 2로 줌으로 제곱 가능 # 디폴트 True
x_train = pf.fit_transform(x_train)
x_test = pf.transform(x_test)

print(x_train.shape)

#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()  
rf = RandomForestClassifier()

model = StackingClassifier(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator= CatBoostClassifier(verbose=0),
    n_jobs=-1, #cpu 전부 사용
    cv = 5,
)

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC : ', accuracy_score(y_test, y_pred))


# model.score :  0.9473684210526315
# 스태킹 ACC :  0.9473684210526315
# stacking만 헀을 때 보다 성능이 낮음.

