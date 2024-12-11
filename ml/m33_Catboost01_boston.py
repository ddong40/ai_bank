import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time 
import random as rn
import tensorflow as tf
tf.random.set_seed(337)
np.random.seed(337)
rn.seed(337)
from catboost import CatBoostClassifier

path = './대회/dacon/생명연구/'

train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# np_path = 'C:/Users/ddong40/ai_2/대회/dacon/생명연구/_save/'


# le_subclass = LabelEncoder()
# train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# x_data = joblib.load(path + 'x_encoded.dat')
# y_data = joblib.load(path + 'y_data.dat')
# test_csv = joblib.load(path + 'test_encoded.dat')

print(train_csv)
print(test_csv)

#########################라벨인코딩##########################
le = LabelEncoder()
y = le.fit_transform(train_csv['SUBCLASS'])

aaa = {}
for i, label in enumerate(le.classes_) : 
    
    aaa[label] = i 
print(aaa)
# {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'COAD': 4, 'DLBC': 5, 'GBMLGG': 6, 'HNSC': 7, 'KIPAN': 8, 'KIRC': 9, 'LAML': 10, 'LGG': 11, 'LIHC': 12,
#  'LUAD': 13, 'LUSC': 14, 'OV': 15, 'PAAD': 16, 'PCPG': 17, 'PRAD': 18, 'SARC': 19, 'SKCM': 20, 'STES': 21, 'TGCT': 22, 'THCA': 23, 'THYM': 24, 'UCEC': 25}

x_data = train_csv.drop(columns=['SUBCLASS', 'ID'])
test_csv = test_csv.drop(columns=['ID'])

print(x_data.shape, y.shape, test_csv.shape) #(6201, 4384) (6201,) (2546, 4384)

categorycal_columns = x_data.select_dtypes(include=['object', 'category']).columns 
print(categorycal_columns) # 컬럼명 뽑기 4384개
print(len(categorycal_columns))

#######################smote####################
x_train, x_test, y_train, y_test = train_test_split(x_data, y, train_size=0.9, random_state=3434)

# smote = SMOTE(random_state=7777)

# x_train, y_train = smote.fit_resample(x_train, y_train)

################################################

parmeters = {
    'learning_rate' : [0.01, 0.03, 0.05, 0.1, 0.2],
    'depth' : [4, 6, 8, 10, 12], # 6,
    'l2_leaf_reg' : [1, 3, 5, 7, 10],
    'bagging_temperature' : [0.0, 0.5, 1.0, 2.0, 5.0],
    'boarder_count' : [32, 64, 128, 255], 
    'random_strenth': [1, 5, 10],
    # 'n_jobs' : -1,  # CPU
}

cat_features = list(range(x_train.shape[1]))
print(cat_features)


#2. 모델
model = CatBoostClassifier(
    iterations = 2000, #트리개수 (기본값 : 500)
    learning_rate=0.1,  # 학습률 (기본값: 0.03)
    depth=6,        # 트리 깊이 (기본값: 6)
    bagging_temperature=1.0, # L2 정규화 (기본값: 3)
    random_state=5, # 랜덤성 추가(기본값 :1)
    border_count=128, #연속형 변수 처리 (기본값 : 254)
    task_type='GPU', 
    devices='0', # 첫 번째 GPU 사용 (기본값 : 모든 GPU 사용)
    early_stopping_rounds=100, #조기종료 (기본값:None)
    verbose=10,  # 10단위로 출력된다는 뜻
    cat_features= cat_features
)

start_time = time.time()
model.fit(x_train, y_train, 
          eval_set=(x_test, y_test)
          )

end_time = time.time()

# 4. 평가, 예측
# score = model.score(x_test, y_test)

# print('model.score : ', model.score())

y_pred = model.predict(x_test)

f1 = f1_score(y_test, y_pred, average='macro')

# predictions = model.predict(test_csv) 

# # 제출용 제작

# # y_predict = model.predict(test_data) # 선생님 코드 

# # original_labels = le_subclass.inverse_transform(predictions)  

# submisson = pd.read_csv(path + "sample_submission.csv")

# submisson["SUBCLASS"] = original_labels

# submisson.to_csv(path + 'baseline_submission.csv', encoding='UTF-8-sig', index=False)    

# # submisson.to_csv(path + f'final_submit_{time.strftime("%Y%m%d_%H%M")}.csv', encoding='UTF-8-sig', index=False)

# print("f1_score : ", f1)
# print("걸린시간: " , round(end_time - start_time, 2), "초")

# # f1_score :  0.3188050043612538
# # 걸린시간:  53.17 초
# # dacon 점수 : 0.28432

