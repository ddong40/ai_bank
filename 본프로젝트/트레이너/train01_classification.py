## tf274gpu 사용 ##

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
from ngboost import NGBClassifier
import pickle
import os

# 랜덤 시드 고정
random_seed = 1186
np.random.seed(random_seed)
random.seed(random_seed)

# 데이터 경로
path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/'

# 데이터 불러오기
df1 = pd.read_csv(path + "z_kaggle_new_landmarks_correct_data.csv", encoding='cp949')
df2 = pd.read_csv(path + "z_kaggle_new_landmarks_incorrect_data.csv", encoding='cp949')

df1['description'] = '정자세'
df2['description'] = '오자세'

# 데이터 결합
df = pd.concat([df1, df2], ignore_index=True)
# 정자세/오자세 이진 분류로 변경
df['description'] = df['description'].map({'정자세': 0, '오자세': 1})

# X, y 분리
x = df.drop(['description'], axis=1).values
y = df['description'].values

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed, stratify=y_train)

# 하이퍼파라미터 설정
n_e = 10000
lr = 0.05
early_stopping_rounds = 100

# 모델 학습
fit_models = {}

# XGBoost 학습
print("\n--- Starting XGBoost training ---")
dtrain_xgb = xgb.DMatrix(x_train, label=y_train)
dval_xgb = xgb.DMatrix(x_val, label=y_val)
dtest_xgb = xgb.DMatrix(x_test)

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': lr,
    'max_depth': 6,
    'tree_method': 'gpu_hist',  # GPU 사용
    'gpu_id': 0                # GPU ID 설정
}

xgb_model = xgb.train(
    params=params_xgb,
    dtrain=dtrain_xgb,
    num_boost_round=n_e,
    evals=[(dval_xgb, 'eval')],
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50  # 50 라운드마다 출력
)
fit_models['xgb'] = xgb_model
print("\n--- XGBoost training completed ---")

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation

params_lgb = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': lr,
    'n_estimators': n_e,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_depth': 6,
    'min_gain_to_split': 0.0,
    'min_data_in_leaf': 20,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# **params_lgb를 통해 딕셔너리를 키워드 인자로 전달
print("\n--- Starting LightGBM training ---")
lgb_model = LGBMClassifier(**params_lgb)

lgb_model.fit(
    x_train,
    y_train,
    eval_set=[(x_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)
fit_models['lgbm'] = lgb_model
print("\n--- LightGBM training completed ---")


# CatBoost 학습
print("\n--- Starting CatBoost training ---")
train_pool = Pool(data=x_train, label=y_train)
val_pool = Pool(data=x_val, label=y_val)

cat_model = CatBoostClassifier(
    iterations=n_e,
    learning_rate=lr,
    depth=6,
    task_type="GPU",
    devices="0",  # GPU 사용
    verbose=50    # 50 라운드마다 출력
)
cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)
fit_models['cat'] = cat_model
print("\n--- CatBoost training completed ---")

# NGBoost 학습
print("\n--- Starting NGBoost training ---")
# 중복 제거 및 스케일링 (필요 시 적용)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

ngb_model = NGBClassifier(verbose=0, n_estimators=1000, learning_rate=lr)

try:
    ngb_model.fit(x_train_scaled, y_train)
    fit_models['ngb'] = ngb_model
    print("\n--- NGBoost training completed ---")
except np.linalg.LinAlgError as e:
    print("NGBoost 학습 중 오류 발생:", e)
print("\n--- NGBoost training completed ---")

# 모델 평가
metrics = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1-score': {}}
for name, model in fit_models.items():
    if name == 'xgb':
        y_pred = xgb_model.predict(dtest_xgb)
        y_pred = (y_pred > 0.5).astype(int)
    elif name == 'lgbm':
        y_pred = lgb_model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int)
    elif name == 'cat':
        y_pred = cat_model.predict(x_test)
    elif name == 'ngb':
        y_pred = ngb_model.predict(x_test)

    metrics['accuracy'][name] = accuracy_score(y_test, y_pred)
    metrics['precision'][name] = precision_score(y_test, y_pred)
    metrics['recall'][name] = recall_score(y_test, y_pred)
    metrics['f1-score'][name] = f1_score(y_test, y_pred)

    print(f"\n--- {name.upper()} 모델 분류 보고서 ---")
    print(classification_report(y_test, y_pred))

# 최종 결과 출력
print("\n--- 최종 평가 결과 ---")
for metric, values in metrics.items():
    print(f"\n{metric.upper()}:")
    for name, score in values.items():
        print(f"{name.upper()}: {score:.4f}")

# 모델 저장
save_path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/가중치/'
os.makedirs(save_path, exist_ok=True)
for name, model in fit_models.items():
    filename = os.path.join(save_path, f"{name}_model.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"{name.upper()} 모델 저장 완료: {filename}")

print("\n모든 모델 저장이 완료되었습니다.")

'''
--- XGB 모델 분류 보고서 ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       411
           1       1.00      0.99      0.99        99

    accuracy                           1.00       510
   macro avg       1.00      0.99      1.00       510
weighted avg       1.00      1.00      1.00       510

[LightGBM] [Warning] min_data_in_leaf is set=20, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=20
[LightGBM] [Warning] feature_fraction is set=0.8, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.8
[LightGBM] [Warning] min_gain_to_split is set=0.0, min_split_gain=0.0 will be ignored. Current value: min_gain_to_split=0.0
[LightGBM] [Warning] bagging_fraction is set=0.8, subsample=1.0 will be ignored. Current value: bagging_fraction=0.8
[LightGBM] [Warning] bagging_freq is set=5, subsample_freq=0 will be ignored. Current value: bagging_freq=5

--- LGBM 모델 분류 보고서 ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       411
           1       1.00      0.99      0.99        99

    accuracy                           1.00       510
   macro avg       1.00      0.99      1.00       510
weighted avg       1.00      1.00      1.00       510


--- CAT 모델 분류 보고서 ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       411
           1       1.00      1.00      1.00        99

    accuracy                           1.00       510
   macro avg       1.00      1.00      1.00       510
weighted avg       1.00      1.00      1.00       510


--- 최종 평가 결과 ---

ACCURACY:
XGB: 0.9980
LGBM: 0.9980
CAT: 1.0000

PRECISION:
XGB: 1.0000
LGBM: 1.0000
CAT: 1.0000

RECALL:
XGB: 0.9899
LGBM: 0.9899
CAT: 1.0000

F1-SCORE:
XGB: 0.9949
LGBM: 0.9949
CAT: 1.0000
'''