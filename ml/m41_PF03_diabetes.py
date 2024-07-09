import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 로지스틱리그레션은 분류
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2134)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pf = PolynomialFeatures(degree=2, include_bias=True) #degree를 2로 줌으로 제곱 가능 # 디폴트 True
x_train = pf.fit_transform(x_train)
x_test = pf.transform(x_test)

#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()
rf = RandomForestRegressor()

model = StackingRegressor(
    estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator = CatBoostRegressor(verbose=0),
    n_jobs = -1,
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('r2 Score : ', r2_score(y_pred, y_test))

# model.score :  0.8456887551162162
# r2 Score :  0.8456887551162162




