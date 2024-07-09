import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 로지스틱리그레션은 분류
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, RandomForestRegressor, StackingRegressor, StackingClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


#1. 데이터
path = "C:/Users/ddong40/ai_2/_data/dacon/diabetes/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['Outcome'], axis=1) 
y = train_csv["Outcome"]

x = x.to_numpy()
x = x/255.

random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=random_state, train_size=0.8,
                                                    stratify=y
                                                    )


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

# model.score :  0.6870229007633588
# 스태킹 ACC :  0.6870229007633588

