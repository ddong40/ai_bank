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
from sklearn.preprocessing import LabelEncoder

import numpy as np
import random

# 랜덤 시드 고정
random_seed = 3333
np.random.seed(random_seed)
random.seed(random_seed)

# xgboost, catboost, ngboost, lightgbm

path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/'


df = pd.read_csv(path + "AI_HUB_landmarks_data.csv", encoding='cp949')
print(df['description'].unique())
df2 = pd.read_csv(path + "Kaggle_landmarks_data.csv", encoding='cp949')
df3 = pd.read_csv(path + "landmarks_data.csv", encoding='cp949')

x = df.drop('description', axis=1).values
y = df['description']

# 다중 클래스 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1186)

n_e = 500
lr = 0.07
# 파이프라인 설정 (OneVsRestClassifier로 감싸 다중 레이블 문제 해결)
pipelines = {
     'xgb': make_pipeline(StandardScaler(), OneVsRestClassifier(XGBClassifier(n_estimators=n_e, learning_rate=lr))),
     'cat': make_pipeline(StandardScaler(), OneVsRestClassifier(CatBoostClassifier(verbose=0,n_estimators=n_e, learning_rate=lr))),
     'ngb': make_pipeline(StandardScaler(), OneVsRestClassifier(NGBClassifier(verbose=0, n_estimators=n_e, learning_rate=lr))),
     'lgbm': make_pipeline(StandardScaler(), OneVsRestClassifier(LGBMClassifier(n_estimators=n_e, learning_rate=lr))),
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

with open(path + 'pushup_test_lgbm_1104.pkl', 'wb') as f:
    pickle.dump(fit_models['lgbm'], f)