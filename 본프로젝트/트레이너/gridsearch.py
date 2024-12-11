import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier  # 다중 레이블 문제 해결
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

# 랜덤 시드 고정
random_seed = 3333
np.random.seed(random_seed)
random.seed(random_seed)

# xgboost, catboost, ngboost, lightgbm

path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/'

df = pd.read_csv(path + "푸시업_전체데이터_1106.csv", encoding='cp949')
print(df['description'].unique())

x = df.drop('description', axis=1).values
y = df['description']

# 다중 클래스 인코딩
le = LabelEncoder()
y = le.fit_transform(y)
# y = pd.get_dummies(y).values


# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4040)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = 0.04
# 파이프라인 설정 (OneVsRestClassifier로 감싸 다중 레이블 문제 해결)
pipelines = {
     'xgb': make_pipeline(StandardScaler(), OneVsRestClassifier(XGBClassifier(n_estimators=400, learning_rate=lr, tree_method='gpu_hist' ))),
     'cat': make_pipeline(StandardScaler(), OneVsRestClassifier(CatBoostClassifier(verbose=0,n_estimators=400, learning_rate=lr, task_type='GPU', devices='0'))),
     'ngb': make_pipeline(StandardScaler(), OneVsRestClassifier(NGBClassifier(verbose=0, n_estimators=400, learning_rate=lr))),
     'lgbm': make_pipeline(StandardScaler(), OneVsRestClassifier(LGBMClassifier(n_estimators=400, learning_rate=lr, device='gpu'))),
}

# 모델 학습
fit_models = {}
for algorithm, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algorithm] = model

# 학습된 모델 확인
print(fit_models)

from sklearn.metrics import classification_report

# 예측 결과를 저장할 딕셔너리 초기화
predictions = {}

# 각 모델에 대해 예측 수행
for algorithm, model in fit_models.items():
    y_pred = model.predict(x_test)  # 테스트 데이터에 대한 예측
    predictions[algorithm] = y_pred  # 예측 결과 저장

# 각 모델의 분류 보고서 출력
for algorithm, y_pred in predictions.items():
    print(f'--- {algorithm} 모델 분류 결과 평가 ---')
    print(classification_report(y_test, y_pred))
    print('-------------------------------')
    
metrics = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1-score': {}
}

for algorithm, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics['accuracy'][algorithm] = accuracy
    metrics['precision'][algorithm] = precision
    metrics['recall'][algorithm] = recall
    metrics['f1-score'][algorithm] = f1

for metric, values in metrics.items():
    print(f'--- {metric} ---')
    for algorithm, score in values.items():
        print(f'{algorithm}: {score:.4f}')

"""
--- accuracy ---
xgb: 0.6562
cat: 0.6582
ngb: 0.3340
lgbm: 0.6875
--- precision ---
xgb: 0.6778
cat: 0.6767
ngb: 0.4147
lgbm: 0.7019
--- recall ---
xgb: 0.6562
cat: 0.6582
ngb: 0.3340
lgbm: 0.6875
--- f1-score ---
xgb: 0.6580
cat: 0.6595
ngb: 0.3446
lgbm: 0.6868
"""


# with open(path + 'pushup_test_lgbm_1104_2.pkl', 'wb') as f:
#     pickle.dump(fit_models['lgbm'], f)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define a parameter grid for LightGBM
param_grid = {
    'onevsrestclassifier__estimator__device': ['gpu'], 
    'onevsrestclassifier__estimator__num_leaves': [31, 63, 127],
    'onevsrestclassifier__estimator__max_depth': [5, 10, -1],
    'onevsrestclassifier__estimator__min_child_samples': [5, 10, 20],
    'onevsrestclassifier__estimator__subsample': [0.8, 1.0],
    'onevsrestclassifier__estimator__colsample_bytree': [0.8, 1.0],
    'onevsrestclassifier__estimator__learning_rate': [0.01, 0.04, 0.1, 0.009, 0.05],  # Add learning rates to try
    'onevsrestclassifier__estimator__n_estimators': [100, 400, 600]       # Add n_estimators to try
}

# Use GridSearchCV to optimize the LGBM pipeline
lgbm_pipeline = pipelines['lgbm']
grid_search = RandomizedSearchCV(
    lgbm_pipeline,
    param_grid,
    scoring='f1_weighted',  # You can change this to optimize for a different metric
    cv=3,                   # 3-fold cross-validation
    verbose=2,
    n_jobs=-1               # Use all available CPU cores
)

# Fit the GridSearch to the training data
grid_search.fit(x_train, y_train)

# Best parameters and best estimator for LightGBM
best_lgbm_model = grid_search.best_estimator_
print("Best parameters for LightGBM:", grid_search.best_params_)
print("Best LightGBM model:", best_lgbm_model)

# Save the best model after tuning
with open(path + 'pushup_optimized_lgbm_grid_search_1108_1.pkl', 'wb') as f:
    pickle.dump(best_lgbm_model, f)

# Evaluate the optimized LightGBM model
y_pred_lgbm = best_lgbm_model.predict(x_test)
print('--- Optimized LightGBM Model Classification Report ---')
print(classification_report(y_test, y_pred_lgbm))

# Calculate and display evaluation metrics for the optimized LightGBM model
accuracy = accuracy_score(y_test, y_pred_lgbm)
precision = precision_score(y_test, y_pred_lgbm, average='weighted')
recall = recall_score(y_test, y_pred_lgbm, average='weighted')
f1 = f1_score(y_test, y_pred_lgbm, average='weighted')

print(f'Optimized LightGBM Model Accuracy: {accuracy:.4f}')
print(f'Optimized LightGBM Model Precision: {precision:.4f}')
print(f'Optimized LightGBM Model Recall: {recall:.4f}')
print(f'Optimized LightGBM Model F1-Score: {f1:.4f}')


# Optimized LightGBM Model Accuracy: 0.6934
# Optimized LightGBM Model Precision: 0.7159
# Optimized LightGBM Model Recall: 0.6934
# Optimized LightGBM Model F1-Score: 0.6922