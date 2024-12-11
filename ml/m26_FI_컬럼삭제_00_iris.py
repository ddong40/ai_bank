### 판다스로 바꿔서 컬럼 삭제 ###
# pd.DataFrame
# 컬럼명 : datasets.feature_names 안에 있지!

# 실습 
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하라
# 데이터셋 재구성 후
# 기존 모델결과와 비교!!! 향상시켜라

from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
#1 데이터

# x, y = load_iris(return_X_y=True)
# datasets = load_iris()
# datasets = pd.DataFrame(datasets)

# Iris 데이터셋 로드
datasets = load_iris()
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)

# Target 변수 추가
y = df['species'] = datasets.target


x = df.drop(columns=['sepal width (cm)', 'species'], axis=1)

print(x.shape)
# print(x)
# print(y)
print(x.shape, y.shape)
# (150, 4) (150,)

import xgboost as xgb


# early_stop = xgb.callback.Earlystopping(
#     rounds = 50,
#     metric_name = 'mlogloss',
#     data_name = 'validation_0',
#     save_best = True
# )



random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123, stratify=y)



#2 모델 구성

parmeters = {
    'n_estimators' : 300, # 1000,
    'alpha': 0.12,
    'colsample_bytree': 0.7,
    'gamma': 2,
    'lambda': 2,
    'learning_rate' : 0.25, # 0.1,
    'max_depth' : 2, # 6,
    'min_child_weight' : 1.9, # 10,
    'tree_method' : 'hist', # GPU / 'gpu_hist'
    'device': 'cuda',
    # 'n_jobs' : -1,  # CPU
}

model = xgb.XGBClassifier(
    parmeters,
    # n_estimators=100,   # n_estimators=100: 이 기계가 100번 반복해서 나무처럼 생각하게 해달라는 뜻, 에포
    # learning_rate=0.1,  # 기계가 배우는 속도
    # max_depth=6,        # 나무의 최대 깊이를 6으로 정해요. 너무 깊어지면 기계가 과하게 배울 수 있어요.
    random_state=42,    
    # use_label_encoder=False, # LabelEncoder를 사용하지 않도록 설정해요. 대신 우리가 직접 인코딩을 했기 때문이에요.
    # eval_metric='mlogloss', # 기계가 얼마나 잘 예측했는지 평가하는 방법을 'mlogloss'로 정해요.
    n_jobs=-1,)

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = False,
          )

print('모델의 점수 : ', model.score(x_test, y_test)) 

y_predict = model.predict(x_test)

# y_pred_best = model.best_estimator_.predict(x_test) 

print('accuracy_score : ', accuracy_score(y_test, y_predict)) 



# 모델의 점수 :  0.9555555555555556
# accuracy_score :  0.9555555555555556