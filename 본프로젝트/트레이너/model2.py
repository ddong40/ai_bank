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

path = 'C:/Users/ddong40/Desktop/본프로젝트/AI Trainer/data/'


df1 = pd.read_csv(path + "AI_HUB_landmarks_data.csv", encoding='utf-8')
# print(df['description'].unique())
df2 = pd.read_csv(path + "Kaggle_landmarks_data.csv", encoding='cp949')
df3 = pd.read_csv(path + "landmarks_data.csv", encoding='cp949')


# # xgboost, catboost, ngboost, lightgbm
# path = 'C:/ai5/본프로젝트/메인프로젝트/data/'
# df1 = pd.read_csv(path + "AI_HUB_landmarks_data.csv", encoding='utf-8')
# df2 = pd.read_csv(path + "Kaggle_landmarks_data.csv", encoding='cp949')

df = pd.concat([df1, df2, df3])

x = df.drop(['description'], axis=1).values
y = df['description']

# 다중 클래스 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1186)

n_e = 5000
lr = 0.05
# 파이프라인 설정 (OneVsRestClassifier로 감싸 다중 레이블 문제 해결)
pipelines = {
     'xgb': make_pipeline(StandardScaler(), XGBClassifier(n_estimators=n_e, learning_rate=lr, tree_method='gpu_hist', gpu_id=0)),
     'cat': make_pipeline(StandardScaler(), CatBoostClassifier(verbose=0,n_estimators=n_e, learning_rate=lr, task_type='GPU', devices='0')),
     'ngb': make_pipeline(StandardScaler(), OneVsRestClassifier(NGBClassifier(verbose=0, n_estimators=n_e, learning_rate=lr))),
     'lgbm': make_pipeline(StandardScaler(), LGBMClassifier(n_estimators=n_e, learning_rate=lr, device='gpu')),
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

with open(path + 'pushup_test_lgbm_1115_1.pkl', 'wb') as f:
    pickle.dump(fit_models['lgbm'], f)

"""
--- accuracy ---
xgb: 0.6958
cat: 0.6369
ngb: 0.3813
lgbm: 0.7266
--- precision ---
xgb: 0.6996
cat: 0.6406
ngb: 0.3913
lgbm: 0.7292
--- recall ---
xgb: 0.6958
cat: 0.6369
ngb: 0.3813
lgbm: 0.7266
--- f1-score ---
xgb: 0.6958
cat: 0.6361
ngb: 0.3743
lgbm: 0.7268

---------------------- 1113 500, 0.05
--- accuracy ---
xgb: 0.7165
cat: 0.6659
ngb: 0.3847
lgbm: 0.7395
--- precision ---
xgb: 0.7201
cat: 0.6696
ngb: 0.3969
lgbm: 0.7433
--- recall ---
xgb: 0.7165
cat: 0.6659
ngb: 0.3847
lgbm: 0.7395
--- f1-score ---
xgb: 0.7163
cat: 0.6655
ngb: 0.3784
lgbm: 0.7398

----------------------------- 1113_2 1000번
-------------------------------
--- accuracy ---
xgb: 0.7280
cat: 0.6926
ngb: 0.3853
lgbm: 0.7496
--- precision ---
xgb: 0.7307
cat: 0.6944
ngb: 0.3973
lgbm: 0.7528
--- recall ---
xgb: 0.7280
cat: 0.6926
ngb: 0.3853
lgbm: 0.7496
--- f1-score ---
xgb: 0.7277
cat: 0.6917
ngb: 0.3795
lgbm: 0.7496
-------------1113_3 2000번 



### 데이터 증강, 5000번 ###
--- accuracy ---
xgb: 0.7651
cat: 0.7615
ngb: 0.4720
lgbm: 0.7846
--- precision ---
xgb: 0.7667
cat: 0.7636
ngb: 0.4730
lgbm: 0.7865
--- recall ---
xgb: 0.7651
cat: 0.7615
ngb: 0.4720
lgbm: 0.7846
--- f1-score ---
xgb: 0.7640
cat: 0.7609
ngb: 0.4645
lgbm: 0.7839

"""