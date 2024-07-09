# new gpu #

import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import Pool, CatBoostClassifier
from ngboost import NGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from lightgbm.callback import early_stopping, log_evaluation

# 데이터 경로
path = 'D:/플랭크/'

# 데이터 불러오기
df1 = pd.read_csv(path + "플랭크_정자세_data2.csv", encoding='cp949')
df2 = pd.read_csv(path + "플랭크_오자세_data2.csv", encoding='cp949')

df3 = df1.head(1000)
df4 = df2.head(1000)

# 데이터 결합 및 레이블 변환
df1['description'] = '정자세'
df2['description'] = '오자세'

# 데이터 결합 및 레이블 변환
df3['description'] = '정자세'
df4['description'] = '오자세'

df_old = pd.concat([df3, df4], ignore_index=True)
df_new = pd.concat([df1, df2], ignore_index=True)
df_old['description'] = df_old['description'].map({'정자세': 0, '오자세': 1})
df_new['description'] = df_new['description'].map({'정자세': 0, '오자세': 1})

# 기존 데이터로 학습
x_old = df_old.drop(['description'], axis=1).values
y_old = df_old['description'].values

# 새로운 데이터
x_new = df_new.drop(['description'], axis=1).values
y_new = df_new['description'].values

def xgb_accuracy(preds, dtrain):
    labels = dtrain.get_label()
    preds_binary = (preds > 0.5).astype(int)
    acc = accuracy_score(labels, preds_binary)
    return ('accuracy', acc)

# Train-Test Split for Old Data
X_train_old, X_val_old, y_train_old, y_val_old = train_test_split(
    x_old, y_old, test_size=0.2, random_state=1186, stratify=y_old
)

# 저장 경로 생성
save_path = 'D:/플랭크/models/'
os.makedirs(save_path, exist_ok=True)

# 모델 학습 및 저장
fit_models = {}

n_estimators = 10000
lr = 0.005
early_stopping_rounds = 100

# --- XGBoost ---
print("\n--- Training XGBoost with Old Data ---")
dtrain = xgb.DMatrix(X_train_old, label=y_train_old)
dval = xgb.DMatrix(X_val_old, label=y_val_old)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': lr,
    'max_depth': 6,
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=n_estimators,
    evals=[(dval, 'validation')],
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50
)
joblib.dump(xgb_model, os.path.join(save_path, 'xgb_model_가중치저장.pkl'))
#xgb_model.save_model(os.path.join(save_path, 'xgb_model_가중치저장.pkl'))
print("--- XGBoost training completed and model saved ---")

# --- LightGBM ---
print("\n--- Training LightGBM with Old Data ---")
lgb_model = LGBMClassifier(
    n_estimators=n_estimators,
    learning_rate=lr,
    max_depth=6,
    device="gpu",  # GPU 사용
    gpu_platform_id=0,
    gpu_device_id=0
)

# Early Stopping을 콜백으로 적용
lgb_model.fit(
    X_train_old,
    y_train_old,
    eval_set=[(X_val_old, y_val_old)],
    callbacks=[
        early_stopping(stopping_rounds=early_stopping_rounds),  # Early Stopping 콜백
        log_evaluation(period=50)  # 로그 출력 주기
    ]
)

joblib.dump(lgb_model, os.path.join(save_path, 'lgbm_model_가중치저장.pkl'))
print("--- LightGBM training completed and model saved ---")

# --- CatBoost ---
print("\n--- Training CatBoost with Old Data ---")
cat_model = CatBoostClassifier(
    iterations=n_estimators,
    learning_rate=lr,
    depth=6,
    task_type="GPU",
    devices="0",
    verbose=50
)
cat_model.fit(
    X_train_old,
    y_train_old,
    eval_set=(X_val_old, y_val_old),
    early_stopping_rounds = early_stopping_rounds
)
joblib.dump(cat_model, os.path.join(save_path, 'cat_model_가중치저장.pkl'))
print("--- CatBoost training completed and model saved ---")

# --- NGBoost ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train_old)
x_test_scaled = scaler.transform(X_val_old)

ngb_model = NGBClassifier(verbose=0, n_estimators=1000, learning_rate=lr)

try:
    ngb_model.fit(x_train_scaled, y_train_old)
    fit_models['ngb'] = ngb_model
    print("\n--- NGBoost training completed ---")
except np.linalg.LinAlgError as e:
    print("NGBoost 학습 중 오류 발생:", e)
joblib.dump(ngb_model, os.path.join(save_path, 'ngb_model_가중치저장.pkl'))
print("--- NGBoost training completed and model saved ---")

# --- Reload and Train with New Data ---
print("\n--- Reloading Models and Training with New Data ---")

# XGBoost 추가 학습
print("\n--- Updating XGBoost with New Data ---")

# Train-Test Split for Old Data
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
    x_new, y_new, test_size=0.2, random_state=1186, stratify=y_new
)

# 모델 로드
xgb_model = joblib.load(os.path.join(save_path, 'xgb_model_가중치저장.pkl'))
print("--- XGBoost model loaded from pkl ---")

dtrain = xgb.DMatrix(X_train_new, label=y_train_new)
dval = xgb.DMatrix(X_val_new, label=y_val_new)

# 새로운 데이터로 추가 학습
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': lr,
    'max_depth': 6,
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=n_estimators,
    evals=[(dval, 'validation')],
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50
)
print("--- XGBoost updated with new data ---")

# LightGBM 추가 학습
print("\n--- Updating LightGBM with New Data ---")
lgb_model = joblib.load(os.path.join(save_path, 'lgbm_model_가중치저장.pkl'))

params_lgb = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': lr,
    'n_estimators': n_estimators,
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

lgb_model.fit(
    X_train_new,
    y_train_new,
    eval_set=[(X_val_new, y_val_new)],
    callbacks=[
        early_stopping(stopping_rounds=early_stopping_rounds),
        log_evaluation(period=50)
    ]
)
print("--- LightGBM updated with new data ---")

# CatBoost 추가 학습
print("\n--- Updating CatBoost with New Data ---")
cat_model = joblib.load(os.path.join(save_path, 'cat_model_가중치저장.pkl'))
cat_model.fit(X_val_new, y_val_new)
print("--- CatBoost updated with new data ---")

# NGBoost 추가 학습
print("\n--- Updating NGBoost with New Data ---")
x_train_scaled2 = scaler.fit_transform(X_train_new)
x_test_scaled2 = scaler.transform(X_val_new)

ngb_model = NGBClassifier(verbose=0, n_estimators=1000, learning_rate=lr)

try:
    ngb_model.fit(x_test_scaled2, y_val_new)
    fit_models['ngb'] = ngb_model
    print("\n--- NGBoost training completed ---")
except np.linalg.LinAlgError as e:
    print("NGBoost 학습 중 오류 발생:", e)

ngb_model = joblib.load(os.path.join(save_path, 'ngb_model_가중치저장.pkl'))

print("--- NGBoost updated with new data ---")

# --- 모델 평가 ---
print("\n--- Evaluating Models ---")
metrics = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1-score': {}}

for name, model in zip(['xgb', 'lgbm', 'cat', 'ngb'], [xgb_model, lgb_model, cat_model, ngb_model]):
    print(f"\n--- Evaluating {name.upper()} model ---")
    if name == 'xgb':
        # XGBoost 모델의 경우 DMatrix로 변환
        dval_new = xgb.DMatrix(X_val_new)
        y_pred = (model.predict(dval_new) > 0.5).astype(int)
    else:
        # 다른 모델(LightGBM, CatBoost, NGBoost)은 numpy 배열 사용
        y_pred = model.predict(X_val_new)
    
    # 성능 평가
    metrics['accuracy'][name] = accuracy_score(y_val_new, y_pred)
    metrics['precision'][name] = precision_score(y_val_new, y_pred)
    metrics['recall'][name] = recall_score(y_val_new, y_pred)
    metrics['f1-score'][name] = f1_score(y_val_new, y_pred)
    print(classification_report(y_val_new, y_pred))


# --- 최종 결과 출력 ---
print("\n--- Final Evaluation Results ---")
for metric, values in metrics.items():
    print(f"\n{metric.upper()}:")
    for name, score in values.items():
        print(f"{name.upper()}: {score:.4f}")
        

# --- 모델 저장 --- #
# 추가 학습된 모델 각각 저장
print("\n--- Saving Updated Models ---")

# XGBoost 모델 저장
joblib.dump(xgb_model, os.path.join(save_path, 'xgb_model_updated.pkl'))  # Booster는 save_model 사용
print("--- XGBoost updated model saved ---")

# LightGBM 모델 저장
joblib.dump(lgb_model, os.path.join(save_path, 'lgbm_model_updated.pkl'))
print("--- LightGBM updated model saved ---")

# CatBoost 모델 저장
joblib.dump(cat_model, os.path.join(save_path, 'cat_model_updated.pkl'))
print("--- CatBoost updated model saved ---")

# NGBoost 모델 저장
joblib.dump(ngb_model, os.path.join(save_path, 'ngb_model_updated.pkl'))
print("--- NGBoost updated model saved ---")


"""
ACCURACY:
XGB: 1.0000
LGBM: 1.0000
CAT: 1.0000
NGB: 0.8199

PRECISION:
XGB: 1.0000
LGBM: 1.0000
CAT: 1.0000
NGB: 0.7829

RECALL:
XGB: 1.0000
LGBM: 1.0000
CAT: 1.0000
NGB: 0.8417

F1-SCORE:
XGB: 1.0000
LGBM: 1.0000
CAT: 1.0000
NGB: 0.8112
"""