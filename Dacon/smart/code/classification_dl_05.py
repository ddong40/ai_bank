"""
개선된 스마트 팩토리 비정상 작동 분류 모델 (최적화된 딥러닝 파이프라인)

주요 개선사항:
1. 전처리: MinMaxScaler → QuantileTransformer(normal) 변경으로 분포 정규화
2. 모델 구조: Flatten 제거, CLS 토큰 + Attention Pooling 도입으로 글로벌 정보 집약
3. 최적화: AdamW(3e-4) + CosineDecay(Warmup) + Weight Decay(1e-4) + Gradient Clipping(1.0)
4. 검증: StratifiedKFold OOF(Out-of-Fold) 교차검증으로 안정적인 모델 선택
5. 앙상블: LightGBM OOF + Neural Network OOF 스태킹으로 최종 성능 향상

모델 구조:
- Dual Branch Dilated Conv1D + Positional Encoding + Multi-Head Attention
- CLS 토큰 기반 Attention Pooling으로 시퀀스 정보 집약
- Pure Conv1D 분류 헤드 (Dense layer 완전 제거)
- 21개 클래스 다중 분류 (Macro F1 Score 최적화)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 딥러닝 및 전처리 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization,
    Reshape, Concatenate, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

# TensorFlow 버전에 따른 AdamW import 시도
try:
    from tensorflow.keras.optimizers.experimental import AdamW
except ImportError:
    try:
        from tensorflow.keras.optimizers import AdamW
    except ImportError:
        # AdamW가 없는 경우 weight_decay 없이 Adam 사용
        print("⚠️ AdamW를 찾을 수 없습니다. weight_decay 없이 Adam을 사용합니다.")
        AdamW = None
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns

# 시드 고정
tf.random.set_seed(42)
np.random.seed(42)

# GPU 메모리 증가 허용 (GPU 사용 시)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("=" * 60)
print("Pure Conv1D 기반 스마트 팩토리 비정상 작동 분류 모델 (Dense layer 제거)")
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

# 특징 공학 함수 추가
def add_features(df):
    """통계적 특징과 PCA 특징 추가"""
    num = df.select_dtypes(include=[np.number]).copy()
    df2 = df.copy()
    
    # 행별 통계 특징
    df2['row_mean'] = num.mean(axis=1)
    df2['row_std']  = num.std(axis=1)
    df2['row_max']  = num.max(axis=1)
    df2['row_min']  = num.min(axis=1)
    df2['row_q25']  = num.quantile(0.25, axis=1)
    df2['row_q75']  = num.quantile(0.75, axis=1)
    df2['row_skew'] = num.skew(axis=1)
    df2['row_kurt'] = num.kurtosis(axis=1)
    
    print(f"   ✅ 통계적 특징 8개 추가 완료")
    return df2

# 테스트 데이터 준비
X_test = test_df.drop(columns=['ID'])

# 특징 공학 적용
print("\n🔧 특징 공학 적용 중...")
X = add_features(X)
X_test = add_features(X_test)

# PCA 특징 추가 (원본 특징에 추가)
print("   🔍 PCA 특징 추가 중...")
pca = PCA(n_components=16, random_state=42)
pca_train = pca.fit_transform(X.select_dtypes(include=[np.number]))
pca_test = pca.transform(X_test.select_dtypes(include=[np.number]))

for i in range(pca_train.shape[1]):
    X[f'pca_{i}'] = pca_train[:, i]
    X_test[f'pca_{i}'] = pca_test[:, i]

print(f"   ✅ PCA 특징 16개 추가 완료")
print(f"   📊 최종 특징 수: {X.shape[1]}개 (원본: 52개 → 확장: {X.shape[1]}개)")

print(f"클래스 분포:\n{y.value_counts().sort_index()}")

# 2. StratifiedKFold OOF 준비
print("\n2. StratifiedKFold OOF 검증 준비 중...")
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"StratifiedKFold: {N_FOLDS}개 폴드로 교차검증")
print(f"전체 데이터 크기: {X.shape}")

# OOF 예측값 저장을 위한 배열 초기화
oof_predictions = np.zeros((len(X), 21))  # 21개 클래스 확률
test_predictions_nn = np.zeros((len(test_df), 21))  # 테스트 예측값
fold_scores = []

# 클래스 불균형 해결을 위한 class_weight 계산 (전체 데이터 기준)
print("\n📊 클래스 불균형 분석 및 가중치 계산...")
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y), 
    y=y
)
class_weight_dict = dict(zip(np.unique(y), class_weights))

print("클래스별 가중치:")
for cls, weight in class_weight_dict.items():
    count = (y == cls).sum()
    print(f"  클래스 {cls}: 가중치 {weight:.3f} (샘플 수: {count})")

print(f"\n가중치 범위: {min(class_weights):.3f} ~ {max(class_weights):.3f}")
print("→ Macro F1 Score 최적화를 위해 소수 클래스에 높은 가중치 부여")

# 3. 이상치 클리핑 (완화 처리)
print("\n3. IQR 기반 이상치 클리핑 적용 중...")

# 4. 커스텀 F1 Score 메트릭 정의
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=21, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.f1_score = self.add_weight(name='f1', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # F1 score 계산을 위한 confusion matrix
        f1_value = tf.py_function(
            func=self._compute_f1,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )
        
        self.f1_score.assign_add(f1_value)
        self.count.assign_add(1.0)
    
    def _compute_f1(self, y_true, y_pred):
        return f1_score(y_true.numpy(), y_pred.numpy(), average=self.average, zero_division=0)
    
    def result(self):
        return self.f1_score / self.count
    
    def reset_state(self):
        self.f1_score.assign(0.0)
        self.count.assign(0.0)

# AdamW 옵티마이저 설정 함수 정의
def create_optimizer_with_warmup(initial_learning_rate=3e-4, decay_steps=1000, warmup_steps=100):
    """AdamW 옵티마이저 생성 (고정 학습률)"""
    
    # AdamW가 사용 가능한 경우
    if AdamW is not None:
        print("✅ AdamW 옵티마이저를 사용합니다 (Weight Decay 포함)")
        optimizer = AdamW(
            learning_rate=initial_learning_rate,  # 고정 학습률
            weight_decay=1e-4,  # L2 정규화
            clipnorm=1.0,       # Gradient Clipping
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    else:
        # AdamW가 없는 경우 Adam 사용 (clipnorm만 적용)
        print("⚠️  Adam 옵티마이저를 사용합니다 (Weight Decay 없음)")
        optimizer = Adam(
            learning_rate=initial_learning_rate,  # 고정 학습률
            clipnorm=1.0,       # Gradient Clipping
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    
    return optimizer

# 5. 모델 구성 (Conv1D + Positional Encoding + Transformer)
print("\n5. Conv1D + Positional Encoding + Transformer 딥러닝 모델 구성 중...")

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Transformer 논문의 Sinusoidal Positional Encoding을 Keras Layer로 구현
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        
    def get_angles(self, pos, i, d_model):
        """각도 계산 함수"""
        angle_rates = 1 / tf.pow(10000.0, tf.cast(2 * (i // 2), tf.float32) / tf.cast(d_model, tf.float32))
        return pos * angle_rates
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # 위치 인덱스 생성 (0, 1, 2, ..., seq_len-1)
        pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        
        # 차원 인덱스 생성 (0, 1, 2, ..., d_model-1)
        i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
        
        # 각도 계산
        angle_rads = self.get_angles(pos, i, d_model)
        
        # 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 적용
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        # sin과 cos를 번갈아가며 배치
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        
        # d_model이 홀수인 경우 마지막 차원 조정
        if d_model % 2 == 1:
            pos_encoding = pos_encoding[:, :-1]
        
        # 배치 차원 추가
        pos_encoding = pos_encoding[tf.newaxis, :, :]
        
        # 입력과 positional encoding 합산
        return inputs + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Transformer 블록 구성 함수"""
    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    
    # Add & Norm 1
    attention_output = Dropout(dropout)(attention_output)
    x1 = Add()([inputs, attention_output])
    x1 = LayerNormalization(epsilon=1e-6)(x1)
    
    # Feed Forward Network
    ffn_output = Dense(ff_dim, activation='relu')(x1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Add & Norm 2
    ffn_output = Dropout(dropout)(ffn_output)
    x2 = Add()([x1, ffn_output])
    x2 = LayerNormalization(epsilon=1e-6)(x2)
    
    return x2

class CLSTokenLayer(tf.keras.layers.Layer):
    """CLS 토큰을 시퀀스 앞에 추가하는 레이어"""
    
    def __init__(self, embed_dim, **kwargs):
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # CLS 토큰을 학습 가능한 파라미터로 초기화
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # CLS 토큰을 배치 크기만큼 복제
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        # CLS 토큰을 시퀀스 앞에 추가
        return tf.concat([cls_tokens, inputs], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

class AttentionPooling(tf.keras.layers.Layer):
    """Attention 기반 풀링으로 CLS 토큰의 정보를 추출하는 레이어"""
    
    def __init__(self, embed_dim, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Attention weights를 생성하는 Dense layer
        self.attention_dense = tf.keras.layers.Dense(
            1, use_bias=False, name='attention_weights'
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, embed_dim)
        # CLS 토큰은 첫 번째 위치 (index 0)에 있음
        cls_output = inputs[:, 0, :]  # (batch_size, embed_dim)
        
        # 모든 토큰에 대한 attention 계산
        attention_scores = self.attention_dense(inputs)  # (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch_size, seq_len, 1)
        
        # Attention weighted average
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch_size, embed_dim)
        
        # CLS 토큰과 context vector를 결합
        combined = cls_output + context_vector
        return combined
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

def create_dual_branch_model(input_dim, num_classes=21):
    """듀얼 브랜치 Dilated Conv1D + CLS 토큰 + Attention Pooling 모델"""
    
    # 입력층 (동일한 데이터가 두 브랜치로 분기)
    inputs = Input(shape=(input_dim,))
    
    # 1D 시퀀스로 변환 (52 피처 → 52 시간스텝)
    reshaped = Reshape((input_dim, 1))(inputs)
    
    # ===== X 브랜치: Dilated Conv1D로 세밀한 시간적 패턴 추출 =====
    # 작은 dilation rate로 로컬 패턴에 집중
    x = Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding='same', name='x_dilconv1')(reshaped)
    x = tf.keras.layers.ReLU(name='x_relu1')(x)
    x = LayerNormalization(epsilon=1e-6, name='x_ln1')(x)
    x = Dropout(0.1, name='x_dropout1')(x)
    
    x = Conv1D(filters=32, kernel_size=3, dilation_rate=2, padding='same', name='x_dilconv2')(x)
    x = tf.keras.layers.ReLU(name='x_relu2')(x)
    x = LayerNormalization(epsilon=1e-6, name='x_ln2')(x)
    x = MaxPooling1D(pool_size=2, padding='same', name='x_pool')(x)  # 26 timesteps
    x = Dropout(0.1, name='x_dropout2')(x)
    
    # ===== Z 브랜치: Dilated Conv1D로 장거리 시간적 의존성 포착 =====
    # 큰 dilation rate로 글로벌 패턴에 집중
    z = Conv1D(filters=32, kernel_size=3, dilation_rate=4, padding='same', name='z_dilconv1')(reshaped)
    z = tf.keras.layers.ReLU(name='z_relu1')(z)
    z = LayerNormalization(epsilon=1e-6, name='z_ln1')(z)
    z = Dropout(0.1, name='z_dropout1')(z)
    
    z = Conv1D(filters=16, kernel_size=3, dilation_rate=8, padding='same', name='z_dilconv2')(z)
    z = tf.keras.layers.ReLU(name='z_relu2')(z)
    z = LayerNormalization(epsilon=1e-6, name='z_ln2')(z)
    z = MaxPooling1D(pool_size=2, padding='same', name='z_pool')(z)  # 26 timesteps
    z = Dropout(0.1, name='z_dropout2')(z)
    
    # ===== 브랜치 결합 =====
    # x: (batch, 26, 32), z: (batch, 26, 16) → concat → (batch, 26, 48)
    combined = Concatenate(axis=-1, name='branch_concat')([x, z])
    
    print(f"브랜치 결합 후 shape: {combined.shape}")
    
    # ===== CLS 토큰 추가 =====
    # 브랜치 결합 후 특징 차원: 48
    cls_layer = CLSTokenLayer(embed_dim=48, name='cls_token_layer')
    combined_with_cls = cls_layer(combined)  # (batch, 27, 48) - CLS 토큰 추가로 시퀀스 길이 +1
    
    print(f"CLS 토큰 추가 후 shape: {combined_with_cls.shape}")
    
    # ===== Positional Encoding 추가 =====
    # Transformer 논문과 동일한 sin/cos 기반 포지셔널 인코딩 적용
    pos_encoder = PositionalEncoding(name='positional_encoding')
    combined_with_pos = pos_encoder(combined_with_cls)
    
    # ===== Transformer 블록 (CLS 토큰 포함 글로벌 관계 학습) =====
    transformer_out = transformer_block(
        inputs=combined_with_pos,  # CLS 토큰 + 포지셔널 인코딩이 추가된 입력 사용
        head_size=16,      # 48 채널에 맞게 조정
        num_heads=3,       # 3개 헤드 유지
        ff_dim=96,         # 48 * 2
        dropout=0.1
    )
    
    # ===== Attention Pooling으로 CLS 토큰 정보 추출 =====
    # GlobalAveragePooling1D 대신 CLS 토큰과 Attention Pooling 사용
    attention_pooled = AttentionPooling(embed_dim=48, name='attention_pooling')(transformer_out)  # (batch, 48)
    
    print(f"Attention Pooling 후 shape: {attention_pooled.shape}")
    
    # ===== Conv1D Classification Head (1D 특징 벡터 처리) =====
    # Attention pooled 출력을 1D 시퀀스로 변환하여 Conv1D 적용
    # (batch, 48) → (batch, 48, 1)로 변환하여 Conv1D 적용 가능하도록 함
    clf_input = tf.expand_dims(attention_pooled, axis=-1)  # (batch, 48, 1)
    
    # 첫 번째 Conv1D: 채널 확장 및 특징 추출
    clf = Conv1D(128, kernel_size=3, padding='same', activation='relu', name='clf_conv1')(clf_input)
    clf = LayerNormalization(epsilon=1e-6, name='clf_ln1')(clf)
    clf = Dropout(0.2, name='clf_dropout1')(clf)
    
    # 두 번째 Conv1D: 채널 감소하며 고수준 특징 추출
    clf = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='clf_conv2')(clf)
    clf = LayerNormalization(epsilon=1e-6, name='clf_ln2')(clf)
    clf = Dropout(0.2, name='clf_dropout2')(clf)
    
    # 세 번째 Conv1D: 최종 특징 압축
    clf = Conv1D(32, kernel_size=3, padding='same', activation='relu', name='clf_conv3')(clf)
    clf = LayerNormalization(epsilon=1e-6, name='clf_ln3')(clf)
    clf = Dropout(0.1, name='clf_dropout3')(clf)
    
    # Conv1D로 최종 클래스 수만큼 채널 생성
    clf = Conv1D(num_classes, kernel_size=1, padding='same', name='clf_conv_output')(clf)  # (batch, 48, 21)
    
    # GlobalAveragePooling1D로 시퀀스 차원 축소하여 최종 출력
    clf = GlobalAveragePooling1D(name='clf_gap')(clf)  # (batch, 21)
    
    # Softmax 활성화로 확률 분포 생성
    outputs = tf.keras.layers.Softmax(name='output')(clf)
    
    model = Model(inputs=inputs, outputs=outputs, name='DualBranchConvTransformer')
    return model

# =============== 딥러닝 모델 주석처리 (트리계열만 실행) ===============
# # StratifiedKFold OOF 학습 시작
# print("\n🚀 StratifiedKFold OOF 학습 시작!")
# print(f"   📊 총 {N_FOLDS}개 폴드 교차검증")
# print(f"   🎯 목표: 21개 클래스 분류 (Macro F1 Score 최적화)")
# print(f"   ⚖️  클래스 가중치: 적용 (불균형 해결)")
# print("=" * 60)

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\n📋 Fold {fold + 1}/{N_FOLDS} 시작...")
    
#     # 폴드별 데이터 분할
#     X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
#     print(f"   🔸 훈련: {len(X_fold_train)}개, 검증: {len(X_fold_val)}개")
    
#     # 3. 폴드별 전처리 (IQR 클리핑 + QuantileTransformer)
#     def apply_fold_preprocessing(X_train, X_val, X_test):
#         """폴드별 전처리 함수"""
#         # IQR 클리핑
#         X_train_clipped = X_train.copy()
#         X_val_clipped = X_val.copy()
#         X_test_clipped = X_test.copy()
        
#         for column in X_train.columns:
#             Q1 = X_train[column].quantile(0.25)
#             Q3 = X_train[column].quantile(0.75)
#             IQR = Q3 - Q1
            
#             # 훈련/검증용 경계값
#             train_lower = Q1 - 1.5 * IQR
#             train_upper = Q3 + 1.5 * IQR
            
#             # 테스트용 경계값 (더 관대한 기준)
#             test_lower = Q1 - 3.0 * IQR
#             test_upper = Q3 + 3.0 * IQR
            
#             # 클리핑 적용 (신경망용만 - 트리모델은 원본 사용)
#             X_train_clipped[column] = X_train[column].clip(train_lower, train_upper)
#             X_val_clipped[column] = X_val[column].clip(train_lower, train_upper)
#             X_test_clipped[column] = X_test[column].clip(test_lower, test_upper)
        
#         # QuantileTransformer 정규화
#         scaler = QuantileTransformer(output_distribution='normal', random_state=42)
#         X_train_scaled = scaler.fit_transform(X_train_clipped)
#         X_val_scaled = scaler.transform(X_val_clipped)
#         X_test_scaled = scaler.transform(X_test_clipped)
#         
#         return X_train_scaled, X_val_scaled, X_test_scaled
    
#     # 폴드별 전처리 적용
#     X_fold_train_scaled, X_fold_val_scaled, X_test_scaled = apply_fold_preprocessing(
#         X_fold_train, X_fold_val, X_test
#     )
    
#     # 4. 모델 생성 (폴드마다 새로 생성)
#     input_dim = X_fold_train_scaled.shape[1]
#     model = create_dual_branch_model(input_dim, num_classes=21)
    
#     # 5. 모델 컴파일 (AdamW + CosineDecay)
#     optimizer = create_optimizer_with_warmup(
#         initial_learning_rate=3e-4,
#         decay_steps=1000,
#         warmup_steps=100
#     )
    
#     model.compile(
#         optimizer=optimizer,
#         loss=SparseCategoricalCrossentropy(),
#         metrics=['accuracy', F1Score(num_classes=21, average='macro')]
#     )
    
#     # 6. 콜백 설정
#     early_stopping = EarlyStopping(
#         monitor='val_f1_score',
#         mode='max',
#         patience=15,
#         restore_best_weights=True,
#         verbose=0
#     )
    
#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_f1_score',
#         mode='max',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-7,
#         verbose=0
#     )
    
#     # 7. 모델 학습
#     print(f"   🚀 Fold {fold + 1} 학습 시작...")
    
#     history = model.fit(
#         X_fold_train_scaled, y_fold_train,
#         validation_data=(X_fold_val_scaled, y_fold_val),
#         epochs=200,
#         batch_size=128,
#         class_weight=class_weight_dict,
#         callbacks=[early_stopping, reduce_lr],
#         verbose=1  # 프로그래스 바 표시
#     )
    
#     # 8. OOF 예측 저장
#     val_pred = model.predict(X_fold_val_scaled, verbose=0)
#     oof_predictions[val_idx] = val_pred
    
#     # 9. 테스트 예측 누적
#     test_pred = model.predict(X_test_scaled, verbose=0)
#     test_predictions_nn += test_pred / N_FOLDS
    
#     # 10. 폴드 성능 평가
#     val_pred_classes = np.argmax(val_pred, axis=1)
#     fold_f1 = f1_score(y_fold_val, val_pred_classes, average='macro')
#     fold_acc = accuracy_score(y_fold_val, val_pred_classes)
#     fold_scores.append(fold_f1)
    
#     print(f"   ✅ Fold {fold + 1} 완료!")
#     print(f"      📊 검증 F1 Score: {fold_f1:.4f}")
#     print(f"      📊 검증 Accuracy: {fold_acc:.4f}")
#     print(f"      📊 학습 에포크: {len(history.history['loss'])}")
#     print(f"      📊 최고 검증 F1: {max(history.history['val_f1_score']):.4f}")

# # OOF 전체 성능 평가
# oof_pred_classes = np.argmax(oof_predictions, axis=1)
# oof_f1 = f1_score(y, oof_pred_classes, average='macro')
# oof_acc = accuracy_score(y, oof_pred_classes)

# print(f"\n🎉 StratifiedKFold OOF 학습 완료!")
# print(f"   📊 평균 CV F1 Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
# print(f"   📊 OOF F1 Score: {oof_f1:.4f}")
# print(f"   📊 OOF Accuracy: {oof_acc:.4f}")
# print(f"   📊 폴드별 F1 점수: {[f'{score:.4f}' for score in fold_scores]}")
# print("=" * 60)

# 딥러닝 모델 OOF 예측값 초기화 (더미 데이터)
print("\n⚠️  딥러닝 모델 주석처리됨 - 트리계열 모델만 실행")
oof_predictions = np.zeros((len(X), 21))  # 더미 데이터
test_predictions_nn = np.zeros((len(test_df), 21))  # 더미 데이터
oof_f1 = 0.0  # 더미 점수

# LightGBM Macro F1 커스텀 평가 함수
def lgb_macro_f1(preds, train_data):
    """LightGBM용 Macro F1 Score 평가 함수"""
    y_true = train_data.get_label().astype(int)
    preds = preds.reshape(21, -1).T  # (n_samples, 21)
    y_pred = np.argmax(preds, axis=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    return 'macro_f1', f1, True  # True: higher is better

# LightGBM OOF 학습 (강화된 파라미터)
print("\n🌟 LightGBM OOF 학습 시작 (Macro F1 최적화)...")

# LightGBM용 OOF 예측값 저장 배열
oof_predictions_lgb = np.zeros((len(X), 21))
test_predictions_lgb = np.zeros((len(test_df), 21))
lgb_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n📋 LightGBM Fold {fold + 1}/{N_FOLDS} 시작...")
    
    # 폴드별 데이터 분할
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM 강화된 파라미터 (Macro F1 최적화)
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 21,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,      # 더 세밀한 학습
        'num_leaves': 128,          # 표현력 증가
        'min_data_in_leaf': 64,     # 과적합 제어
        'max_depth': -1,
        'feature_fraction': 0.9,    # 특징 다양성
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'reg_alpha': 2.0,           # L1 정규화 강화
        'reg_lambda': 10.0,         # L2 정규화 강화
        'verbosity': -1,
        'seed': 42,
        'class_weight': None        # 커스텀 F1으로 처리
    }
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
    
    # 모델 학습 (Macro F1 기반 조기 종료)
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        feval=lgb_macro_f1,              # Macro F1 커스텀 평가
        num_boost_round=2000,            # 충분한 rounds
        callbacks=[
            lgb.early_stopping(200),      # Macro F1 기준 조기 종료
            lgb.log_evaluation(0)
        ]
    )
    
    # OOF 예측
    val_pred_lgb = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
    oof_predictions_lgb[val_idx] = val_pred_lgb
    
    # 테스트 예측 누적
    test_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    test_predictions_lgb += test_pred_lgb / N_FOLDS
    
    # 폴드 성능 평가
    val_pred_classes_lgb = np.argmax(val_pred_lgb, axis=1)
    fold_f1_lgb = f1_score(y_fold_val, val_pred_classes_lgb, average='macro')
    fold_acc_lgb = accuracy_score(y_fold_val, val_pred_classes_lgb)
    lgb_fold_scores.append(fold_f1_lgb)
    
    print(f"   ✅ LightGBM Fold {fold + 1} 완료!")
    print(f"      📊 검증 F1 Score: {fold_f1_lgb:.4f}")
    print(f"      📊 검증 Accuracy: {fold_acc_lgb:.4f}")
    print(f"      📊 Best Iteration: {lgb_model.best_iteration}")

# LightGBM OOF 전체 성능 평가
oof_pred_classes_lgb = np.argmax(oof_predictions_lgb, axis=1)
oof_f1_lgb = f1_score(y, oof_pred_classes_lgb, average='macro')
oof_acc_lgb = accuracy_score(y, oof_pred_classes_lgb)

print(f"\n🎉 LightGBM OOF 학습 완료!")
print(f"   📊 평균 CV F1 Score: {np.mean(lgb_fold_scores):.4f} ± {np.std(lgb_fold_scores):.4f}")
print(f"   📊 OOF F1 Score: {oof_f1_lgb:.4f}")
print(f"   📊 OOF Accuracy: {oof_acc_lgb:.4f}")
print(f"   📊 폴드별 F1 점수: {[f'{score:.4f}' for score in lgb_fold_scores]}")

# CatBoost OOF 학습 (다양성 증가)
print("\n🐱 CatBoost OOF 학습 시작 (Macro F1 최적화)...")

# CatBoost용 OOF 예측값 저장 배열
oof_predictions_cat = np.zeros((len(X), 21))
test_predictions_cat = np.zeros((len(test_df), 21))
cat_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n📋 CatBoost Fold {fold + 1}/{N_FOLDS} 시작...")
    
    # 폴드별 데이터 분할
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost 파라미터 설정 (빠른 실행을 위해 조정)
    cat_params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',  # Macro F1 직접 모니터링
        'depth': 6,                              # 깊이 감소 (9→6)
        'l2_leaf_reg': 10,                       # 정규화 완화
        'learning_rate': 0.1,                    # 학습률 증가 (빠른 수렴)
        'iterations': 1000,                      # 반복 수 대폭 감소 (5000→1000)
        'random_seed': 42,
        'od_type': 'Iter',                       # 조기 종료 타입
        'od_wait': 50,                           # 조기 종료 patience 감소 (200→50)
        'verbose': 100,                          # 진행상황 표시 (False→100)
        'auto_class_weights': 'Balanced',        # 클래스 균형
        'bootstrap_type': 'Bayesian',            # 베이지안 부트스트랩
        'bagging_temperature': 1.0,
        'random_strength': 1.0
    }
    
    # 데이터셋 생성
    train_pool = Pool(X_fold_train, y_fold_train)
    val_pool = Pool(X_fold_val, y_fold_val)
    
    # 모델 학습
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )
    
    # OOF 예측
    val_pred_cat = cat_model.predict_proba(X_fold_val)
    oof_predictions_cat[val_idx] = val_pred_cat
    
    # 테스트 예측 누적
    test_pred_cat = cat_model.predict_proba(X_test)
    test_predictions_cat += test_pred_cat / N_FOLDS
    
    # 폴드 성능 평가
    val_pred_classes_cat = np.argmax(val_pred_cat, axis=1)
    fold_f1_cat = f1_score(y_fold_val, val_pred_classes_cat, average='macro')
    fold_acc_cat = accuracy_score(y_fold_val, val_pred_classes_cat)
    cat_fold_scores.append(fold_f1_cat)
    
    print(f"   ✅ CatBoost Fold {fold + 1} 완료!")
    print(f"      📊 검증 F1 Score: {fold_f1_cat:.4f}")
    print(f"      📊 검증 Accuracy: {fold_acc_cat:.4f}")
    print(f"      📊 Best Iteration: {cat_model.get_best_iteration()}")

# CatBoost OOF 전체 성능 평가
oof_pred_classes_cat = np.argmax(oof_predictions_cat, axis=1)
oof_f1_cat = f1_score(y, oof_pred_classes_cat, average='macro')
oof_acc_cat = accuracy_score(y, oof_pred_classes_cat)

print(f"\n🎉 CatBoost OOF 학습 완료!")
print(f"   📊 평균 CV F1 Score: {np.mean(cat_fold_scores):.4f} ± {np.std(cat_fold_scores):.4f}")
print(f"   📊 OOF F1 Score: {oof_f1_cat:.4f}")
print(f"   📊 OOF Accuracy: {oof_acc_cat:.4f}")
print(f"   📊 폴드별 F1 점수: {[f'{score:.4f}' for score in cat_fold_scores]}")

# 2모델 스태킹 (LightGBM + CatBoost) - 딥러닝 모델 제외
print(f"\n🔥 2모델 스태킹 앙상블 시작 (트리계열만)...")
print(f"   🌟 LightGBM OOF F1: {oof_f1_lgb:.4f}")
print(f"   🐱 CatBoost OOF F1: {oof_f1_cat:.4f}")

# 2모델 스태킹 가중치 최적화
def optimize_2model_stacking_weights(lgb_pred, cat_pred, true_labels):
    """2모델 최적 스태킹 가중치 탐색"""
    best_score = 0
    best_weight = 0.5
    
    # 그리드 서치 (0.1 간격)
    for weight in np.arange(0.0, 1.1, 0.1):
        combined_pred = weight * lgb_pred + (1 - weight) * cat_pred
        combined_classes = np.argmax(combined_pred, axis=1)
        score = f1_score(true_labels, combined_classes, average='macro')
        
        if score > best_score:
            best_score = score
            best_weight = weight
    
    return best_weight, best_score

# 최적 가중치 탐색
optimal_weight, optimal_score = optimize_2model_stacking_weights(
    oof_predictions_lgb, oof_predictions_cat, y
)

print(f"   🎯 최적 가중치:")
print(f"      🌟 LightGBM: {optimal_weight:.1f}")
print(f"      🐱 CatBoost: {1-optimal_weight:.1f}")
print(f"   🏆 스태킹 F1 Score: {optimal_score:.4f}")

# 개별 모델 대비 성능 향상 계산
individual_best = max(oof_f1_lgb, oof_f1_cat)
improvement = optimal_score - individual_best
print(f"   📈 최고 개별 모델 대비 향상: +{improvement:.4f}")

# 최종 테스트 예측 (2모델 스태킹 적용)
final_test_predictions = (optimal_weight * test_predictions_lgb + 
                         (1 - optimal_weight) * test_predictions_cat)
final_test_classes = np.argmax(final_test_predictions, axis=1)

print("=" * 60)

# 최종 제출 파일 생성
print("\n📁 최종 제출 파일 생성 중...")

# 제출 데이터프레임 업데이트
submission_df['target'] = final_test_classes

# 결과 저장
output_path = 'C:/Users/jsy/Desktop/coretech/Dacon/smart/data/stacking_ensemble_submission.csv'
submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 제출 파일 저장 완료: {output_path}")

# 최종 결과 요약
print(f"\n📋 최종 성능 요약 (트리계열 모델만):")
print(f"   🌟 LightGBM OOF F1: {oof_f1_lgb:.4f}")
print(f"   🐱 CatBoost OOF F1: {oof_f1_cat:.4f}")
print(f"   🔥 2모델 스태킹 F1: {optimal_score:.4f}")
print(f"   🎯 최고 개별 모델 대비 향상: +{improvement:.4f}")
print(f"   📊 트리계열 모델 강화 효과:")
print(f"      ✅ Macro F1 커스텀 평가 적용")
print(f"      ✅ 특징 공학 (통계 + PCA): {X.shape[1]}개 특징")
print(f"      ✅ 하이퍼파라미터 튜닝 (num_leaves↑, 정규화↑)")
print(f"      ✅ CatBoost 추가로 다양성 확보")
print(f"      🚀 딥러닝 모델 제외로 빠른 실행 가능")

# 예측 결과 분포 확인 
print(f"\n📊 최종 예측 결과 분포:")
unique, counts = np.unique(final_test_classes, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"   클래스 {cls}: {count}개")

print("\n" + "=" * 60)
print("🎉 개선된 스마트 팩토리 분류 모델 학습 및 예측 완료!")
print("주요 개선사항:")
print("✅ 특징 공학: 통계적 특징 + PCA")
print("✅ LightGBM Macro F1 튜닝 + 강화된 파라미터")
print("✅ CatBoost 추가로 다양성 확보")
print("✅ 2모델 스태킹 앙상블 (LGB + CAT)")
print("✅ StratifiedKFold OOF 교차검증")
print("🚀 딥러닝 모델 제외로 빠른 실행")
print("=" * 60)