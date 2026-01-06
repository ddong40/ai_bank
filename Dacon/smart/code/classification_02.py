"""
LightGBM 기반 다중 클래스 분류 모델
- IQR 기반 이상치 클리핑으로 완화 처리
- Stratified Split으로 라벨 균형 유지
- LightGBM으로 21개 클래스 분류
- 하이퍼파라미터 튜닝 및 교차 검증 적용
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 시드 고정
np.random.seed(42)

print("=" * 60)
print("LightGBM 기반 스마트 팩토리 비정상 작동 분류 모델")
print("=" * 60)

# 1. 데이터 적재 및 분리
print("\n1. 데이터 적재 중...")
train_df = pd.read_csv('C:/Users/jsy/Desktop/coretech/Dacon/smart/data/train.csv')
test_df = pd.read_csv('C:/Users/jsy/Desktop/coretech/Dacon/smart/data/test.csv')
submission_df = pd.read_csv('C:/Users/jsy/Desktop/coretech/Dacon/smart/data/sample_submission.csv')

# X, y 분리
X = train_df.drop(columns=['target', 'ID'])
y = train_df['target']  # 정수 라벨 그대로 유지 (0~20)

print(f"훈련 데이터 크기: {X.shape}")
print(f"피처 수: {X.shape[1]}")
print(f"클래스 수: {len(y.unique())}")
print(f"클래스 분포:\n{y.value_counts().sort_index()}")

# 2. Stratified Split (라벨 균형 유지)
print("\n2. 데이터 분할 중...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 세트: {X_train.shape[0]}개")
print(f"검증 세트: {X_val.shape[0]}개")

# 3. 이상치 클리핑 (완화 처리)
print("\n3. IQR 기반 이상치 클리핑 적용 중...")

def apply_iqr_clipping(X_train, X_val, X_test):
    """IQR 기반으로 이상치를 클리핑하는 함수"""
    X_train_clipped = X_train.copy()
    X_val_clipped = X_val.copy()
    X_test_clipped = X_test.copy()
    
    clip_info = {}
    
    for column in X_train.columns:
        # 훈련 데이터에서 IQR 계산
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # 경계값 설정
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        clip_info[column] = (lower_bound, upper_bound)
        
        # 모든 데이터셋에 클리핑 적용
        X_train_clipped[column] = X_train[column].clip(lower_bound, upper_bound)
        X_val_clipped[column] = X_val[column].clip(lower_bound, upper_bound)
        X_test_clipped[column] = X_test[column].clip(lower_bound, upper_bound)
    
    return X_train_clipped, X_val_clipped, X_test_clipped, clip_info

# 테스트 데이터 준비
X_test = test_df.drop(columns=['ID'])

# 클리핑 적용
X_train_clipped, X_val_clipped, X_test_clipped, clip_info = apply_iqr_clipping(
    X_train, X_val, X_test
)

print(f"클리핑 완료. 처리된 피처 수: {len(clip_info)}")

# 4. 정규화 (MinMax Scaling)
print("\n4. MinMaxScaler를 이용한 정규화 중...")
scaler = MinMaxScaler()

# 훈련 데이터에 fit, 모든 데이터에 transform (0-1 범위로 스케일링)
X_train_scaled = scaler.fit_transform(X_train_clipped)
X_val_scaled = scaler.transform(X_val_clipped)
X_test_scaled = scaler.transform(X_test_clipped)

print("MinMax 정규화 완료 (범위: 0-1)")

# 5. LightGBM 모델 구성 및 하이퍼파라미터 설정
print("\n5. LightGBM 모델 구성 중...")

# LightGBM 기본 하이퍼파라미터 설정
lgb_params = {
    'objective': 'multiclass',
    'num_class': 21,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

print("LightGBM 하이퍼파라미터:")
for key, value in lgb_params.items():
    print(f"  {key}: {value}")

# 입력 차원 확인
input_dim = X_train_scaled.shape[1]
print(f"\n입력 차원: {input_dim}")
print(f"클래스 수: 21")

# 6. 교차 검증 설정
print("\n6. Stratified K-Fold 교차 검증 설정 중...")

# Stratified K-Fold 설정 (클래스 불균형을 고려)
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"교차 검증: {n_folds}-Fold Stratified K-Fold")
print("각 폴드마다 클래스 분포가 균등하게 유지됩니다.")

# 7. LightGBM 모델 학습 및 교차 검증
print("\n7. LightGBM 모델 교차 검증 학습 시작...")

print(f"\n🚀 LightGBM 모델 학습 시작!")
print(f"   📈 교차 검증: {n_folds}-Fold")
print(f"   🎯 목표: 21개 클래스 분류 (최고 F1 스코어)")
print(f"   🔧 모델: LightGBM Gradient Boosting")
print(f"   🌟 특징: 효율적이고 빠른 그래디언트 부스팅")
print("=" * 60)

# 교차 검증 결과 저장용
cv_scores = []
cv_f1_scores = []
oof_predictions = np.zeros(len(X_train_scaled))  # Out-of-fold 예측값
test_predictions = np.zeros((len(X_test_scaled), 21))  # 테스트 예측값 (확률)

fold_num = 1
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    print(f"\n📊 Fold {fold_num}/{n_folds} 학습 중...")
    
    # 데이터 분할
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
    
    # 모델 학습
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,  # 최대 1000 라운드 (딥러닝의 epoch와 유사)
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),  # 100 라운드 개선 없으면 중단
            lgb.log_evaluation(period=50)  # 50 라운드마다 성능 출력
        ]
    )
    
    # 검증 데이터 예측
    val_pred_proba = model.predict(X_fold_val, num_iteration=model.best_iteration)
    val_pred_classes = np.argmax(val_pred_proba, axis=1)
    
    # Out-of-fold 예측값 저장
    oof_predictions[val_idx] = val_pred_classes
    
    # 테스트 데이터 예측 (평균을 위해 누적)
    test_pred_proba = model.predict(X_test_scaled, num_iteration=model.best_iteration)
    test_predictions += test_pred_proba / n_folds
    
    # 성능 평가
    fold_accuracy = accuracy_score(y_fold_val, val_pred_classes)
    fold_f1 = f1_score(y_fold_val, val_pred_classes, average='macro')
    
    cv_scores.append(fold_accuracy)
    cv_f1_scores.append(fold_f1)
    
    print(f"   ✅ Fold {fold_num} 완료:")
    print(f"      📈 정확도: {fold_accuracy:.4f}")
    print(f"      🎯 F1 점수: {fold_f1:.4f}")
    print(f"      🌟 최적 부스팅 라운드: {model.best_iteration}")
    
    fold_num += 1

# 전체 교차 검증 결과
print(f"\n🎉 교차 검증 완료!")
print(f"   📊 평균 정확도: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"   🎯 평균 F1 점수: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
print(f"   🏆 최고 정확도: {max(cv_scores):.4f}")
print(f"   💫 최고 F1 점수: {max(cv_f1_scores):.4f}")
print("=" * 60)

# 8. Out-of-Fold 예측 평가
print("\n8. Out-of-Fold 예측 평가 중...")

# Out-of-fold 예측 평가 (전체 훈련 데이터에 대한 교차 검증 예측)
oof_accuracy = accuracy_score(y_train, oof_predictions)
oof_f1 = f1_score(y_train, oof_predictions, average='macro')

print(f"Out-of-Fold 정확도: {oof_accuracy:.4f}")
print(f"Out-of-Fold F1 점수: {oof_f1:.4f}")

# 분류 리포트
print("\nOut-of-Fold 분류 리포트:")
print(classification_report(y_train, oof_predictions))

# 교차 검증 결과 시각화
plt.figure(figsize=(15, 5))

# 폴드별 성능 시각화
plt.subplot(1, 3, 1)
folds = range(1, n_folds + 1)
plt.bar(folds, cv_scores, alpha=0.7, label='Accuracy', color='skyblue')
plt.axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.subplot(1, 3, 2)
plt.bar(folds, cv_f1_scores, alpha=0.7, label='F1 Score', color='lightgreen')
plt.axhline(np.mean(cv_f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_f1_scores):.4f}')
plt.title('Cross-Validation F1 Score')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.legend()

# Out-of-Fold 혼동 행렬
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_train, oof_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Out-of-Fold Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('C:/Users/jsy/Desktop/coretech/Dacon/smart/model/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 테스트 데이터 예측 (교차 검증 앙상블)
print("\n9. 테스트 데이터 예측 중...")

# 테스트 데이터 예측 (교차 검증으로 이미 계산됨)
test_pred_classes = np.argmax(test_predictions, axis=1)

print(f"테스트 예측 완료: {len(test_pred_classes)}개 샘플")
print("교차 검증 앙상블 예측을 사용합니다.")

# 10. 제출 파일 생성
print("\n10. 제출 파일 생성 중...")

submission_df['target'] = test_pred_classes

# 결과 저장
output_path = 'C:/Users/jsy/Desktop/coretech/Dacon/smart/data/deeplearning_submission.csv'
submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"제출 파일 저장 완료: {output_path}")

# 예측 결과 분포 확인
print(f"\n예측 결과 분포:")
unique, counts = np.unique(test_pred_classes, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"클래스 {cls}: {count}개")

print("\n" + "=" * 60)
print("LightGBM 분류 모델 학습 및 예측 완료!")
print("=" * 60)