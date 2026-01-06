"""
LightGBM 기반 다중 클래스 분류 모델 (F1 최적화 버전)
- IQR 기반 이상치 클리핑으로 완화 처리
- 전처리 단순화: 스케일링/정규화 제거 (트리 모델 특성)
- Macro F1 직접 최적화 및 모니터링
- 사용자 정의 F1 평가 함수로 early stopping
- 장기 학습 & 높은 규제로 과적합 방지
- Stratified Split으로 라벨 균형 유지
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 시드 고정
np.random.seed(42)

print("=" * 60)
print("LightGBM 기반 스마트 팩토리 비정상 작동 분류 모델 (F1 최적화)")
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

# 2. 사용자 정의 Macro F1 평가 함수 정의
print("\n2. 사용자 정의 Macro F1 평가 함수 정의...")

def lgb_macro_f1_eval(y_pred, y_true):
    """LightGBM용 Macro F1 평가 함수"""
    y_true = y_true.get_label()
    y_pred = y_pred.reshape(21, -1).T
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true, y_pred_classes, average='macro')
    return 'macro_f1', f1, True  # (eval_name, eval_result, is_higher_better)

print("사용자 정의 Macro F1 평가 함수 정의 완료")

# 3. 데이터 분할 (Stratified Split으로 라벨 균형 유지)
print("\n3. 데이터 분할 중...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 세트: {X_train.shape[0]}개")
print(f"검증 세트: {X_val.shape[0]}개")

# 4. 이상치 클리핑 (완화 처리)
print("\n4. IQR 기반 이상치 클리핑 적용 중...")

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

# 최종 데이터 (이상치 처리됨, 스케일링은 제거)
X_train_final = X_train_clipped
X_test_final = X_test_clipped

print(f"최종 훈련 데이터: {X_train_final.shape}")
print(f"최종 테스트 데이터: {X_test_final.shape}")
print("전처리: IQR 이상치 클리핑만 적용 (스케일링/정규화는 트리 모델 특성상 제거)")

# 4. LightGBM 모델 구성 및 하이퍼파라미터 설정 (F1 최적화)
print("\n4. LightGBM 모델 구성 중 (F1 최적화 설정)...")

# LightGBM F1 최적화 하이퍼파라미터 설정 (충분한 훈련)
lgb_params = {
    'objective': 'multiclass',
    'num_class': 21,
    'metric': 'None',  # 사용자 정의 평가 함수 사용
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.03,  # 러닝레이트 더 감소 (0.05 → 0.03) - 더 세밀한 학습
    'feature_fraction': 0.8,  # 규제 강화 (0.9 → 0.8)
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.05,  # L1 규제 완화 (0.1 → 0.05) - 더 긴 훈련 허용
    'lambda_l2': 0.05,  # L2 규제 완화 (0.1 → 0.05) - 더 긴 훈련 허용
    'min_data_in_leaf': 15,  # 과적합 방지 완화 (20 → 15)
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

print("LightGBM 하이퍼파라미터:")
for key, value in lgb_params.items():
    print(f"  {key}: {value}")

# 입력 차원 확인
input_dim = X_train_final.shape[1]
print(f"\n입력 차원: {input_dim}")
print(f"클래스 수: 21")
print("F1 최적화 설정: 러닝레이트 0.03, 최대 5000 라운드, 500 라운드 인내심")

# 5. 교차 검증 설정
print("\n5. Stratified K-Fold 교차 검증 설정 중...")

# Stratified K-Fold 설정 (클래스 불균형을 고려)
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"교차 검증: {n_folds}-Fold Stratified K-Fold")
print("각 폴드마다 클래스 분포가 균등하게 유지됩니다.")

# 6. LightGBM 모델 학습 및 교차 검증 (F1 최적화)
print("\n6. LightGBM 모델 교차 검증 학습 시작 (F1 최적화)...")

print(f"\n🚀 LightGBM F1 최적화 모델 학습 시작!")
print(f"   📈 교차 검증: {n_folds}-Fold")
print(f"   🎯 목표: Macro F1 스코어 직접 최적화")
print(f"   🔧 모델: LightGBM (사용자 정의 F1 평가)")
print(f"   🌟 특징: IQR 이상치 클리핑 + F1 기준 early stopping")
print(f"   ⚙️ 설정: 학습률 0.03 + 최대 5000 라운드 + 500 인내심")
print(f"   🏃 훈련: 충분한 훈련을 위한 긴 학습 설정")
print("=" * 60)

# 교차 검증 결과 저장용
cv_scores = []
cv_f1_scores = []
oof_predictions = np.zeros(len(X_train_final))  # Out-of-fold 예측값
test_predictions = np.zeros((len(X_test_final), 21))  # 테스트 예측값 (확률)

fold_num = 1
for train_idx, val_idx in skf.split(X_train_final, y_train):
    print(f"\n📊 Fold {fold_num}/{n_folds} 학습 중...")
    
    # 데이터 분할
    X_fold_train, X_fold_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
    
    # 모델 학습 (F1 최적화 - 충분한 훈련)
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=5000,  # 부스팅 라운드 대폭 증가 (2000 → 5000)
        feval=lgb_macro_f1_eval,  # 사용자 정의 F1 평가 함수
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=True),  # 더 긴 인내심 (200 → 500)
            lgb.log_evaluation(period=100)  # 100 라운드마다 성능 출력
        ]
    )
    
    # 검증 데이터 예측
    val_pred_proba = model.predict(X_fold_val, num_iteration=model.best_iteration)
    val_pred_classes = np.argmax(val_pred_proba, axis=1)
    
    # Out-of-fold 예측값 저장
    oof_predictions[val_idx] = val_pred_classes
    
    # 테스트 데이터 예측 (평균을 위해 누적)
    test_pred_proba = model.predict(X_test_final, num_iteration=model.best_iteration)
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

# 7. Out-of-Fold 예측 평가 (F1 최적화 결과)
print("\n7. Out-of-Fold 예측 평가 중 (F1 최적화 결과)...")

# Out-of-fold 예측 평가 (전체 훈련 데이터에 대한 교차 검증 예측)
oof_accuracy = accuracy_score(y_train, oof_predictions)
oof_f1 = f1_score(y_train, oof_predictions, average='macro')

print(f"Out-of-Fold 정확도: {oof_accuracy:.4f}")
print(f"Out-of-Fold Macro F1 점수: {oof_f1:.4f}")

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
plt.title('Out-of-Fold Confusion Matrix (F1 Optimized)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('C:/Users/jsy/Desktop/coretech/Dacon/smart/model/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 테스트 데이터 예측 (F1 최적화 앙상블)
print("\n8. 테스트 데이터 예측 중 (F1 최적화 앙상블)...")

# 테스트 데이터 예측 (교차 검증으로 이미 계산됨)
test_pred_classes = np.argmax(test_predictions, axis=1)

print(f"테스트 예측 완료: {len(test_pred_classes)}개 샘플")
print("F1 최적화 교차 검증 앙상블 예측을 사용합니다.")

# 9. 제출 파일 생성
print("\n9. 제출 파일 생성 중...")

submission_df['target'] = test_pred_classes

# 결과 저장
output_path = 'C:/Users/jsy/Desktop/coretech/Dacon/smart/data/f1_optimized_submission.csv'
submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"F1 최적화 제출 파일 저장 완료: {output_path}")

# 예측 결과 분포 확인
print(f"\n예측 결과 분포:")
unique, counts = np.unique(test_pred_classes, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"클래스 {cls}: {count}개")

print("\n" + "=" * 60)
print("LightGBM F1 최적화 분류 모델 학습 및 예측 완료!")
print("주요 개선사항:")
print("- IQR 이상치 클리핑 적용 (스케일링/정규화는 제거)")
print("- Macro F1 직접 최적화 (사용자 정의 평가 함수)")
print("- F1 기준 early stopping")
print("- 충분한 훈련: 학습률 0.03 + 최대 5000 라운드 + 500 인내심")
print("=" * 60)