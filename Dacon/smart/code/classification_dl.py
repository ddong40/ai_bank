"""
Conv1D + Transformer 기반 다중 클래스 분류 모델
- IQR 기반 이상치 클리핑으로 완화 처리
- Stratified Split으로 라벨 균형 유지
- Conv1D(2층) + Sinusoidal Positional Encoding + Multi-Head Attention Transformer로 21개 클래스 분류
- Transformer 논문과 동일한 cos/sin 기반 포지셔널 인코딩 적용
- EarlyStopping 및 ReduceLROnPlateau 적용
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
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
print("Conv1D + Transformer 기반 스마트 팩토리 비정상 작동 분류 모델")
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
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"훈련 세트: {X_train.shape[0]}개")
print(f"검증 세트: {X_val.shape[0]}개")

# 클래스 불균형 해결을 위한 class_weight 계산 (Macro F1 Score 최적화)
print("\n📊 클래스 불균형 분석 및 가중치 계산...")
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("클래스별 가중치:")
for cls, weight in class_weight_dict.items():
    count = (y_train == cls).sum()
    print(f"  클래스 {cls}: 가중치 {weight:.3f} (샘플 수: {count})")

print(f"\n가중치 범위: {min(class_weights):.3f} ~ {max(class_weights):.3f}")
print("→ Macro F1 Score 최적화를 위해 소수 클래스에 높은 가중치 부여")

# 3. 이상치 클리핑 (완화 처리)
print("\n3. IQR 기반 이상치 클리핑 적용 중...")

def apply_iqr_clipping(X_train, X_val, X_test, test_clipping=True, test_multiplier=3.0):
    """
    IQR 기반으로 이상치를 클리핑하는 함수
    
    Args:
        test_clipping: 테스트 데이터에 클리핑 적용 여부
        test_multiplier: 테스트 데이터용 IQR 배수 (더 관대한 기준)
    """
    X_train_clipped = X_train.copy()
    X_val_clipped = X_val.copy()
    X_test_clipped = X_test.copy()
    
    clip_info = {}
    
    for column in X_train.columns:
        # 훈련 데이터에서 IQR 계산
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # 훈련/검증용 경계값 (일반적인 1.5 * IQR)
        train_lower = Q1 - 1.5 * IQR
        train_upper = Q3 + 1.5 * IQR
        
        # 테스트용 경계값 (더 관대한 기준)
        test_lower = Q1 - test_multiplier * IQR
        test_upper = Q3 + test_multiplier * IQR
        
        clip_info[column] = {
            'train_bounds': (train_lower, train_upper),
            'test_bounds': (test_lower, test_upper)
        }
        
        # 훈련/검증 데이터 클리핑 (기존과 동일)
        X_train_clipped[column] = X_train[column].clip(train_lower, train_upper)
        X_val_clipped[column] = X_val[column].clip(train_lower, train_upper)
        
        # 테스트 데이터 클리핑 (선택적 적용)
        if test_clipping:
            X_test_clipped[column] = X_test[column].clip(test_lower, test_upper)
            
    return X_train_clipped, X_val_clipped, X_test_clipped, clip_info

# 테스트 데이터 준비
X_test = test_df.drop(columns=['ID'])

# 클리핑 적용 (테스트 데이터는 더 관대한 기준 적용)
X_train_clipped, X_val_clipped, X_test_clipped, clip_info = apply_iqr_clipping(
    X_train, X_val, X_test, 
    test_clipping=True,      # 테스트 데이터 클리핑 적용
    test_multiplier=3.0      # 1.5 대신 3.0 * IQR (더 관대한 기준)
)

print(f"클리핑 완료. 처리된 피처 수: {len(clip_info)}")
print(f"📋 클리핑 설정:")
print(f"   🔸 훈련/검증 데이터: 1.5 * IQR 기준 (엄격)")
print(f"   🔸 테스트 데이터: 3.0 * IQR 기준 (관대) - 정보 손실 최소화")

# 클리핑된 데이터 비교
train_outliers = (X_train != X_train_clipped).sum().sum()
test_outliers = (X_test != X_test_clipped).sum().sum()
print(f"   📊 클리핑된 값 수 - 훈련: {train_outliers}개, 테스트: {test_outliers}개")

# 4. 정규화 (MinMax Scaling)
print("\n4. MinMaxScaler를 이용한 정규화 중...")
scaler = MinMaxScaler()

# 훈련 데이터에 fit, 모든 데이터에 transform (0-1 범위로 스케일링)
X_train_scaled = scaler.fit_transform(X_train_clipped)
X_val_scaled = scaler.transform(X_val_clipped)
X_test_scaled = scaler.transform(X_test_clipped)

print("MinMax 정규화 완료 (범위: 0-1)")

# 정규화 상태 검증
print(f"\n📋 정규화 상태 검증:")
print(f"   🔸 훈련 데이터 범위: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
print(f"   🔸 검증 데이터 범위: [{X_val_scaled.min():.3f}, {X_val_scaled.max():.3f}]")
print(f"   🔸 테스트 데이터 범위: [{X_test_scaled.min():.3f}, {X_test_scaled.max():.3f}]")
print(f"   ✅ 모든 데이터가 동일한 스케일러로 정규화됨")

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

def create_dual_branch_model(input_dim, num_classes=21):
    """듀얼 브랜치 Dilated Conv1D + Transformer 모델 (x, z 브랜치)"""
    
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
    
    # ===== Positional Encoding 추가 =====
    # Transformer 논문과 동일한 sin/cos 기반 포지셔널 인코딩 적용
    pos_encoder = PositionalEncoding(name='positional_encoding')
    combined_with_pos = pos_encoder(combined)
    
    # ===== Transformer 블록 (결합된 특징으로 글로벌 관계 학습) =====
    transformer_out = transformer_block(
        inputs=combined_with_pos,  # 포지셔널 인코딩이 추가된 입력 사용
        head_size=16,      # 48 채널에 맞게 조정
        num_heads=3,       # 3개 헤드 유지
        ff_dim=96,         # 48 * 2
        dropout=0.1
    )
    
    # ===== Flatten for Direct Dense Connection =====
    # Transformer 출력을 직접 Dense층으로 연결 (GAP/MaxPool 제거)
    flattened = tf.keras.layers.Flatten(name='flatten')(transformer_out)
    
    # ===== Classification Head =====
    clf = Dense(128, activation='relu', name='clf_dense1')(flattened)
    clf = LayerNormalization(epsilon=1e-6, name='clf_ln1')(clf)
    clf = Dropout(0.2, name='clf_dropout1')(clf)
    
    clf = Dense(64, activation='relu', name='clf_dense2')(clf)
    clf = LayerNormalization(epsilon=1e-6, name='clf_ln2')(clf)
    clf = Dropout(0.2, name='clf_dropout2')(clf)
    
    # 출력층
    outputs = Dense(num_classes, activation='softmax', name='output')(clf)
    
    model = Model(inputs=inputs, outputs=outputs, name='DualBranchConvTransformer')
    return model

def create_ensemble_compatible_model(input_dim, num_classes=21, model_id=0):
    """앙상블용 다양한 아키텍처 모델"""
    inputs = Input(shape=(input_dim,))
    
    if model_id == 0:  # Dilated Conv1D 기본 모델
        x = Reshape((input_dim, 1))(inputs)
        x = Conv1D(32, 3, dilation_rate=1, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        x = Conv1D(64, 3, dilation_rate=3, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = PositionalEncoding()(x)  # Positional Encoding 추가
        x = transformer_block(x, 16, 2, 128, 0.1)
        x = tf.keras.layers.Flatten()(x)  # GAP 대신 Flatten 사용
        
    elif model_id == 1:  # 더 깊은 Dilated Conv1D
        x = Reshape((input_dim, 1))(inputs)
        x = Conv1D(24, 3, dilation_rate=1, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        x = Conv1D(48, 3, dilation_rate=2, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        x = Conv1D(96, 3, dilation_rate=4, padding='same', activation='relu')(x)
        x = LayerNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)  # GAP 대신 Flatten 사용
    
    # 공통 분류층
    x = Dense(128, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# 듀얼 브랜치 모델 생성 (x, z 브랜치로 다양한 특징 추출)
input_dim = X_train_scaled.shape[1]
model = create_dual_branch_model(input_dim, num_classes=21)

# 커스텀 F1 Score 메트릭 정의
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

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy', F1Score(num_classes=21, average='macro')]
)

print("모델 구조:")
model.summary()

# 6. 학습 설정 (콜백)
print("\n6. 학습 콜백 설정 중...")

# 커스텀 모니터링 콜백
class DetailedMonitoringCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        val_f1 = logs.get('val_f1_score', 0.0)
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')
        train_f1 = logs.get('f1_score', 0.0)
        
        # 현재 학습률 가져오기
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        print(f"\n📊 Epoch {epoch + 1} 결과:")
        print(f"   🔸 Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1: {train_f1:.4f}")
        print(f"   🔸 Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        print(f"   🔸 Learning Rate: {current_lr:.2e}")
        
        # 검증 F1 점수 개선 체크 (주요 모니터링 지표)
        if val_f1 > self.best_val_f1:
            improvement = val_f1 - self.best_val_f1
            self.best_val_f1 = val_f1
            self.epochs_without_improvement = 0
            print(f"   ✅ 검증 F1 점수 개선! (+{improvement:.4f}) 🎯")
        else:
            self.epochs_without_improvement += 1
            print(f"   ⚠️  검증 F1 점수 개선 없음 ({self.epochs_without_improvement}회 연속)")
        
        # 검증 손실 개선 체크
        if val_loss < self.best_val_loss:
            improvement = self.best_val_loss - val_loss
            self.best_val_loss = val_loss
            print(f"   ✅ 검증 손실 개선! (이전 대비 -{improvement:.4f})")
        
        # 검증 정확도 개선 체크
        if val_accuracy > self.best_val_accuracy:
            improvement = val_accuracy - self.best_val_accuracy
            self.best_val_accuracy = val_accuracy
            print(f"   ✅ 검증 정확도 개선! (+{improvement:.4f})")
        
        print("-" * 60)

# EarlyStopping: 검증 F1 점수가 개선되지 않으면 조기 종료
early_stopping = EarlyStopping(
    monitor='val_f1_score',
    mode='max',  # F1 점수는 높을수록 좋음
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: 검증 F1 점수가 개선되지 않으면 학습률 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_score',
    mode='max',  # F1 점수는 높을수록 좋음
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 모니터링 콜백
monitoring_callback = DetailedMonitoringCallback()

callbacks = [early_stopping, reduce_lr, monitoring_callback]

# 7. 모델 학습
print("\n7. 모델 학습 시작...")

print(f"\n🚀 듀얼 브랜치 모델 학습 시작!")
print(f"   📈 총 에포크: 200 (최대)")
print(f"   📊 배치 크기: 128")
print(f"   🎯 목표: 21개 클래스 분류 (Macro F1 Score 최적화)")
print(f"   ⚖️  클래스 가중치: 적용 (불균형 해결)")
print(f"   ⏱️  조기 종료: 15 에포크 개선 없으면 중단")
print(f"   🔧 모델 구조: X 브랜치(Dilated 1,2) + Z 브랜치(Dilated 4,8) + Transformer")
print(f"   🌟 특징: 듀얼 Dilated Conv1D → Positional Encoding → Multi-Head Attention")
print("=" * 60)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=128,
    class_weight=class_weight_dict,  # Macro F1 Score 최적화를 위한 클래스 가중치 적용
    callbacks=callbacks,
    verbose=0  # 커스텀 콜백이 출력을 담당하므로 0으로 설정
)

print(f"\n🎉 학습 완료!")
print(f"   📊 총 {len(history.history['loss'])} 에포크 실행")
print(f"   🏆 최종 검증 정확도: {max(history.history['val_accuracy']):.4f}")
print(f"   🎯 최종 검증 F1 점수: {max(history.history['val_f1_score']):.4f}")
print(f"   💫 최종 검증 손실: {min(history.history['val_loss']):.4f}")
print("=" * 60)

# 8. 모델 평가
print("\n8. 모델 평가 중...")

# 검증 데이터 예측
y_val_pred = model.predict(X_val_scaled)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# 정확도 및 Macro F1 Score 계산
val_accuracy = accuracy_score(y_val, y_val_pred_classes)
val_macro_f1 = f1_score(y_val, y_val_pred_classes, average='macro')
val_weighted_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')

print(f"🎯 대회 평가 지표 (Macro F1 Score): {val_macro_f1:.4f}")
print(f"📊 검증 정확도: {val_accuracy:.4f}")
print(f"⚖️  Weighted F1 Score: {val_weighted_f1:.4f}")

# 분류 리포트 (클래스별 성능 상세 분석)
print("\n📋 클래스별 성능 분석:")
print(classification_report(y_val, y_val_pred_classes))

# 학습 곡선 시각화 (Macro F1 Score 포함)
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(history.history['f1_score'], label='Training F1 (Macro)')
plt.plot(history.history['val_f1_score'], label='Validation F1 (Macro)')
plt.title('Macro F1 Score (대회 평가 지표)')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# 혼동 행렬
plt.subplot(1, 4, 4)
cm = confusion_matrix(y_val, y_val_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('C:/Users/jsy/Desktop/coretech/Dacon/smart/model/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 테스트 데이터 예측 (이미 학습된 모델 사용)
print("\n9. 테스트 데이터 예측 중...")

# 테스트 데이터 예측 (이미 학습된 최적 모델 사용)
test_predictions = model.predict(X_test_scaled)
test_pred_classes = np.argmax(test_predictions, axis=1)

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
print("Conv1D + Transformer 분류 모델 학습 및 예측 완료!")
print("=" * 60)