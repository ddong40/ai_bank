#new gpu
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 로지스틱리그레션은 분류
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(y)
print(np.unique(y))

print(x.shape, y.shape)
# (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=3434)

#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()  
rf = RandomForestRegressor()

model = VotingRegressor(
    estimators= [('XGB', xgb), ('RF', rf), ('CAT', cat)],
    # voting ='soft', # soft는 n분의 1
    # voting ='hard', # 디폴트 # hard는 다수결
    )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)

print('acc score : ', acc)

# 최종점수 :  0.8445206366575616
# acc score :  0.8445206366575616

