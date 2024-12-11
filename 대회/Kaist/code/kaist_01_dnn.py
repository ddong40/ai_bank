import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
# from sklearn.metrics import 

# print(plt.__version__) #3.9.2
print(sns.__version__) #0.13.2
# python --version 3.9.7



import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./study/대회/Kaist/data/경진대회용 주조 공정최적화 데이터셋.csv', encoding='cp949') #cp949사용으로 한글 깨짐 방지

train.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 92015 entries, 0 to 92014
# Data columns (total 32 columns):
#  #   Column                        Non-Null Count  Dtype  
# ---  ------                        --------------  -----
#  0   Unnamed: 0                    92015 non-null  int64
#  1   line                          92015 non-null  object
#  2   name                          92015 non-null  object
#  3   mold_name                     92015 non-null  object
#  4   time                          92015 non-null  object
#  5   date                          92015 non-null  object
#  6   count                         92015 non-null  int64
#  7   working                       92014 non-null  object
#  8   emergency_stop                92014 non-null  object
#  9   molten_temp                   89754 non-null  float64
#  10  facility_operation_cycleTime  92015 non-null  int64
#  11  production_cycletime          92015 non-null  int64
#  12  low_section_speed             92014 non-null  float64
#  13  high_section_speed            92014 non-null  float64
#  14  molten_volume                 46885 non-null  float64
#  15  cast_pressure                 92014 non-null  float64
#  16  biscuit_thickness             92014 non-null  float64
#  17  upper_mold_temp1              92014 non-null  float64
#  18  upper_mold_temp2              92014 non-null  float64
#  19  upper_mold_temp3              91702 non-null  float64
#  20  lower_mold_temp1              92014 non-null  float64
#  21  lower_mold_temp2              92014 non-null  float64
#  22  lower_mold_temp3              91702 non-null  float64
#  23  sleeve_temperature            92014 non-null  float64
#  24  physical_strength             92014 non-null  float64
#  25  Coolant_temperature           92014 non-null  float64
#  26  EMS_operation_time            92015 non-null  int64
#  27  registration_time             92015 non-null  object
#  28  passorfail                    92014 non-null  float64
#  29  tryshot_signal                1919 non-null   object
#  30  mold_code                     92015 non-null  int64
#  31  heating_furnace               42869 non-null  object
# dtypes: float64(16), int64(6), object(10)
# memory usage: 22.5+ MB

print(train.isnull().sum())

# Unnamed: 0                          0
# line                                0
# name                                0
# mold_name                           0
# time                                0
# date                                0
# count                               0
# working                             1
# emergency_stop                      1
# molten_temp                      2261
# facility_operation_cycleTime        0
# production_cycletime                0
# low_section_speed                   1
# high_section_speed                  1
# molten_volume                   45130
# cast_pressure                       1
# biscuit_thickness                   1
# upper_mold_temp1                    1
# upper_mold_temp2                    1
# upper_mold_temp3                  313
# lower_mold_temp1                    1
# lower_mold_temp2                    1
# lower_mold_temp3                  313
# sleeve_temperature                  1
# physical_strength                   1
# Coolant_temperature                 1
# EMS_operation_time                  0
# registration_time                   0
# passorfail                          1
# tryshot_signal                  90096
# mold_code                           0
# heating_furnace                 49146

# 공통행 제거
# train = train.drop(train[train['working'].isnull()].index)

#1. 데이터 전처리

print(train['line'].value_counts())

# 전자교반 3라인 2호기    92015
# Name: line, dtype: int64

print(train['name'].value_counts())
# TM Carrier RH    92015
# Name: name, dtype: int64

# a = []
# for i in range(1, 355):
#     a.append(len(train[train['count']]==i))
# # import koreanize_matplotlib

# sns.lineplot(a)
# plt.title('count 별 행의 수')
# plt.xlabel('count')
# plt.ylabel('행의 수')
# plt.show()

print('-------------------')

# working
print(train['working'].unique()) 
# ['가동' '정지' nan]

# train = train['working'].fillna('Nan')

# train = train[train['working'].isna()]

# sns.boxplot(train['molten_temp']) #박스 플롯 보고 싶을 때 이거 쓰셈

#series.quantile(0.25)는 1사분위수
#series.quantile(0.5)는 중앙값
#series.quantile(0.75)는 3사분위수
q25 = train['molten_temp'].quantile(0.25) - (train['molten_temp'].quantile(0.75) - train['molten_temp'].quantile(0.25))
print(q25) 
#696.0 

# 이상치 처리 25% 알류미늄은 660도 정도에서 녹음
train['molten_temp'] = np.where(train['molten_temp'] < q25, q25, train['molten_temp'])
# np.where(조건, x, y)
# 'molten_temp'가 q25보다 작으면 q25, 크면 원값 유지

q75 = train['molten_temp'].quantile(0.75) + (train['molten_temp'].quantile(0.75) - train['molten_temp'].quantile(0.25))
print(q75)
#747.0

print(train['molten_temp'].isnull().sum())
#2261

#컬럼이 결측치(NaN)인 행만을 필터링하여 tempna_df에 저장
tempna_df = train[train['molten_temp'].isnull()]
print(tempna_df['passorfail'].value_counts)


print(tempna_df.head(2))
#      Unnamed: 0          line           name                        mold_name        time      date  ...  EMS_operation_time    registration_time passorfail  tryshot_signal  mold_code  heating_furnace
# 11895       11895  전자교반 3라인 2호기  TM Carrier RH  TM Carrier RH-Semi-Solid DIE-06  2019-01-19  01:20:45  ...            23  2019-01-19 01:20:45        0.0             NaN       8412              NaN      
# 11897       11897  전자교반 3라인 2호기  TM Carrier RH  TM Carrier RH-Semi-Solid DIE-06  2019-01-19  01:22:35  ...            23  2019-01-19 01:22:35        0.0             NaN       8412              NaN  
# 연속적 결측 발생

from scipy.interpolate import CubicSpline

# 결측치 행 선택
missing_rows = train['molten_temp'].isnull()

# 결측치가 아닌 행의 인덱스와 값 추출
known_indexs = train.index[~missing_rows]
known_values = train.loc[~missing_rows, 'molten_temp']

# 결측치가 아닌 행의 인덱스와 값을 리스트로 변환
known_indexs_list = known_indexs.tolist()
known_values_list = known_values.tolist()

# CubioSpline 객체 생성
cs = CubicSpline(known_indexs_list, known_values_list)

# 결측치를 보간한 값으로 채워주기
train.loc[missing_rows, 'molten_temp'] = cs(train.loc[missing_rows].index)

# 결측치 확인
print(train['molten_temp'].isnull().sum())
# 0

# <사이클시간>
# 112초를 기준으로 불량률에 차이가 있음

print(train.groupby(['passorfail']).agg(설비사이클시간평균 = ('facility_operation_cycleTime', 'mean')))

#              설비사이클시간평균
# passorfail
# 0.0         121.568411
# 1.0         121.291584

q1 = train['facility_operation_cycleTime'].quantile(0.25)
q3 = train['facility_operation_cycleTime'].quantile(0.75)
IQR = q3 - q1
q25 = q1 - IQR*1.5
q75 = q3 + IQR*1.5
print(q25, q75)
# 109.5 129.5

# 사분위수 보다 작을 때에 불량률
a = train[train['facility_operation_cycleTime']<q25]
a['passorfail'].value_counts()
b = a['passorfail'].value_counts().values[1] / sum(a['passorfail'].value_counts().values)*100

#.values는 pandas를 넘파이 형태로 반환, values[1] 은 value_counts로 반환한 값들 중에서 두번째 배열을 나타낸다. 즉 1

# 사분위수 보다 클 때에 불량률
a = train[train['facility_operation_cycleTime']>=q25]
a['passorfail'].value_counts()
c = a['passorfail'].value_counts().values[1] / sum(a['passorfail'].value_counts().values)*100

labels = ['109.5 초 미만', '109.5초 이상']
values = [b,c]

# # 그래프 그리기
# plt.bar(labels, values)
# for i, v in enumerate(values):
#     plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
# # 그래프 제목과 축 레이블 설정
# plt.title('설비 작동 사이클 시간에 따른 불량률')
# plt.xlabel('설비 작동 사이클 시간')
# plt.ylabel('불량률', labelpad=20, rotation=0)

# plt.show() # 막대그래프 보고싶을 때 이거 쓰셈


# <제품 생산 사이클 시간> 
# 115초 기준으로 불량률 차이가 있음

print(train['production_cycletime'].unique())

print(train.groupby(['passorfail']).agg(설비사이클시간평균 = ('production_cycletime', 'mean')))
# passorfail
# 0.0         122.737062
# 1.0         117.668825


#. groupby() 는 panas 데이터프레임의 그룹별로 뒤에 따라올 집계함수를 실행해주는 것이다. 즉 passorfail의 항목에서 같은 그룹 끼리 무언가를 해줌
# 즉 0과 1 기준으로 그룹화를 진행한 뒤 각 그룹에 대해 production _cycletime 열의 평균을 계산하여 설비 사이클 시간 평균이라는 변수로 반환한다.

 
# sns.boxplot(train['production_cycletime'])
# plt.show()

q1 = train['production_cycletime'].quantile(0.25)
q3 = train['production_cycletime'].quantile(0.75)
IQR = q3 - q1
q25 = q1 - IQR*1.5
q75 = q3 + IQR*1.5
print(q25, q75)
# 113.0 129.0

a = train[train['production_cycletime']>=q25]
a['passorfail'].value_counts()
c = a['passorfail'].value_counts().values[1] / sum(a['passorfail'].value_counts().values)*100
print(c) 
# 3.8290714803984085

a = train[train['production_cycletime']<q25]
a['passorfail'].value_counts()
d = a['passorfail'].value_counts().values[1] / sum(a['passorfail'].value_counts().values)*100

print(d)
# 49.334600760456276

labels = ['113초(25%) 이하', '113초(25%) 이상']
values = [d, c]

# 그래프

# plt.bar(labels, values)
# for i, v in enumerate(values):
#     plt.text(i, v, f"{v:.2f}%", ha='center', va='bottom')

# # 그래프 제목과 축 레이블 생성

# plt.title('생산 사이클 시간에 따른 불량률')
# plt.xlabel('생산 사이클 시간')
# plt.ylabel('불량률', labelpad=20, rotation=0)

# plt.show()

train[train['production_cycletime']==0]['passorfail'].value_counts()
print('제품생산 사이클 시간이 0인 데이터의 불량률 : ', train[train['production_cycletime']==0]['passorfail'].value_counts().values[0]/sum(train[train['production_cycletime']==0]['passorfail'].value_counts())*100)

train[train['production_cycletime']<=115]['production_cycletime'].mean()

train['production_cycletime'] = np.where(train['production_cycletime']==0, train[train['production_cycletime']<=115]['production_cycletime'].mean(), train['production_cycletime'])

# <low_section_speed(저속구간속도)>

print(train['low_section_speed'].describe())

# count    92014.000000
# mean       110.794999
# std        305.181962
# min          0.000000
# 25%        110.000000
# 50%        110.000000
# 75%        110.000000
# max      65535.000000
# Name: low_section_speed, dtype: float64

#100 미만일 때 불량률 : 
# 10 이하일 때 불량률 : 

ax110 = train[train['low_section_speed']<10]
print(ax110['passorfail'].value_counts())

# passorfail
# 1.0    301
# 0.0     18

print('저속 구간 10미만일 때 불량률 : ', ax110['passorfail'].value_counts().values[1] / sum(ax110['passorfail'].value_counts().values)*100)
a = ax110['passorfail'].value_counts().values[1]/sum(ax110['passorfail'].value_counts().values)*100
# 저속 구간 10미만일 때 불량률 :  5.6426332288401255

ax110 = train[(train['low_section_speed']<100)&(train['low_section_speed']>=10)]
print(ax110['passorfail'].value_counts())

b = ax110['passorfail'].value_counts().values[1]/sum(ax110['passorfail'].value_counts().values)*100
print('저속 구간 10이상 100미만 : ', b)
# 저속 구간 10이상 100미만 :  47.704081632653065

ax110 = train[(train['low_section_speed']>150)]
# print(ax110['passorfail'].value_counts().values[0])
c = ax110['passorfail'].value_counts().values[0]/sum(ax110['passorfail'].value_counts().values)*100
print('저속 구간 150 이상 : ', c)
# 저속 구간 150 이상 :  100.0

labels = ['~10(미만)', '10(이상)~100(미만)', '100(이상)~150(이하)', '150(이상)~']
values = [a, b, c, d]

# fig = plt.figure(figsize=(8,4))

# # 그래프 그리기
# plt.bar(labels, values)
# for i, v in enumerate(values):
#     plt.text(i, v, f"{v:.2f}%", ha='center', va='bottom')
    
# # 그래프 제목 축 레이블 설정
# plt.title('저속구간속도에 따른 불량률')
# plt.xlabel('저속구간속도', labelpad=10)
# plt.ylabel('불량률', labelpad=20, rotation=0)

# plt.show()

# 이상치

train['low_section_speed'] = np.where(train['low_section_speed']>=60000, 150, train['low_section_speed'])

# <high_section_speed(고속구간속도)>

# 90을 기준으로 불량률이 다름
print(train['high_section_speed'].describe())

# count    92014.000000
# mean       112.624959
# std         10.759272
# min          0.000000
# 25%        112.000000
# 50%        112.000000
# 75%        112.000000
# max        388.000000
# Name: high_section_speed, dtype: float64

# sns.histplot(train['high_section_speed'])
# plt.show()

ax110 = train[(train['high_section_speed']<90)]
ax110['passorfail'].value_counts()

a = ax110['passorfail'].value_counts().values[0]/sum(ax110['passorfail'].value_counts().values)*100
print('고속구간속도가 90미만일 때 불량률 : ', a)
# 고속구간속도가 90미만일 때 불량률 :  100.0

ax110 = train[(train['high_section_speed']>90)]

b = ax110['passorfail'].value_counts().values[1]/sum(ax110['passorfail'].value_counts().values)*100

print('고속구간속도가 90 이상일 때 불량률 :', b)
# 고속구간속도가 90 이상일 때 불량률 : 4.3376165938294955

# <molten_volume(용탕량)>
# 이상치는 양불과 관련이 없다.
# 결측치 제거
# 생략!

# <주조압력>
# 주조 압력에 따른 불량률 변화가 큼
# 중요한 정보
print(train['cast_pressure'].isnull().sum())


# <비스켓 두께>

# <상금형 온도>
# 100보다 낮을수록 불량률 낮아짐
# 1400도 넘는 것 한개이므로 삭제

train = train.drop(train[train['upper_mold_temp1']>1400].index, axis=0)

# <upper2> 
# 4000이 넘는 이상치 1개 행 제거
# 80을 기준으로 양품이 더 많아짐
# 80보다 작아질수록 불량률 증가

train = train.drop(train[train['upper_mold_temp2']>4000].index, axis=0)

#<upper3>

# 결측치처리
# upper 1,2 와 다르게 양불 없음

# 결측치 있는 행 선택
missing_rows = train['upper_mold_temp3'].isnull()

# 결측치가 아닌 행의 인덱스와 값 추출
known_indexes = train.index[~missing_rows]
known_values = train.loc[~missing_rows, 'upper_mold_temp3']

# 결측치가 아닌 행의 인덱스와 값을 리스트로 변환
known_indexes_list = known_indexes.tolist()
known_values_list = known_values.tolist()

# CubioSpline 객체 생성
cs = CubicSpline(known_indexes_list, known_values_list)

# 결측치 보간한 값으로 채워주기
train.loc[missing_rows, 'upper_mold_temp3'] = cs(train.loc[missing_rows].index)

# <lower mold temp(하금형 온도)>

# 온도가 내려갈수록 불량룰 발생

# 60000이 넘는 이상치 최빈값으로 대체

train.loc[train[train['lower_mold_temp3']>2000].index, 'lower_mold_temp3'] = train['lower_mold_temp3'].mode()

# 결측치가 있는 행 선택
missing_rows = train['lower_mold_temp3'].isnull()

# 결측치가 아닌 행의 인덱스와 값 추출
known_indexes = train.index[~missing_rows]
known_values = train.loc[~missing_rows, 'lower_mold_temp3']

# 결측치가 아닌 행의 인덱스와 값을 리스트로 변환
known_indexes_list = known_indexes.tolist()
known_values_list = known_values.tolist()

# CubicSpline 객체 생성
cs = CubicSpline(known_indexes_list, known_values_list)

# 결측치를 보간한 값으로 채워주기
train.loc[missing_rows, 'lower_mold_temp3'] = cs(train.loc[missing_rows].index)

# <phsical_strength 형체력>

train['physical_strength'] = np.where(train['physical_strength']>=60000, 
                                      train[(train['passorfail']==1) & (train['mold_code']==8412)]['physical_strength'].median(), train['physical_strength'])

# <Coolant_temperature 냉각수 온도>

# 따로 처리할 건 없음, 근데 중요한 컬럼

# <mold_code(금형코드)

# count가 7 미만이면 공정 조건이 비슷하지만
# 7 초과이면 공정 조건이 금형 코드마다 다름
# 금형마다 불량률이 비슷함
# 따라서, 불량의 원인은 금형이 아님


####  열 삭제 ####

remove_col = ['working', 'Unnamed: 0', 'line', 'name', 'mold_name', 'time', 'date', 'emergency_stop',
              'molten_volume', 'tryshot_signal', 'heating_furnace', 'registration_time']
train.drop(remove_col, axis=1 , inplace=True)

#### 범주화 ####
train_notcat = train.copy()
train_cat = train.copy()

# #### 비 범주화 #### 
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# train_notcat['working'] = le.fit_transform(train_notcat['working'])

#### 범주화 진행 #### 
cat_cols = ['facility_operation_cycleTime', 'production_cycletime', 'high_section_speed',
           'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'lower_mold_temp1']

threshold = [109.5, 113, 90, 300, 62, 80, 80]

def categorize_col(df, cat_cols, threshold):
    for i in range(len(cat_cols)):
        df[cat_cols[i]] = np.where(df[cat_cols[i]]<threshold[i], 0, 1)
    df['low_section_speed'] = np.where(df['low_section_speed']<10, 0,
                                       np.where(df['low_section_speed']<100, 1,
                                                np.where(df['low_section_speed']<150, 2, 3)))
    df['upper_mold_temp2'] = np.where(df['upper_mold_temp2']<80, 0,
                                     np.where(df['upper_mold_temp2']<239, 1, 2))
    df['lower_mold_temp2'] = np.where(df['lower_mold_temp2']<86.5, 0,
                                     np.where(df['lower_mold_temp2']<314.5, 1, 2))
    
    return df
train_cat = categorize_col(train_cat, cat_cols=cat_cols, threshold=threshold)

print(train_cat.head())
#    count working  molten_temp  facility_operation_cycleTime  production_cycletime  low_section_speed  ...  sleeve_temperature  physical_strength  Coolant_temperature  EMS_operation_time  passorfail  mold_code
# 0    258      가동        731.0                             1                     1                  2  ...               550.0              700.0                 34.0                  23         0.0       8722      
# 1    243      가동        720.0                             0                     1                  2  ...               481.0                0.0                 30.0                  25         0.0       8412      
# 2    244      가동        721.0                             0                     1                  2  ...               481.0                0.0                 30.0                  25         0.0       8412      
# 3    245      가동        721.0                             0                     1                  2  ...               483.0                0.0                 30.0                  25         0.0       8412      
# 4    246      가동        721.0                             0                     1                  2  ...               486.0                0.0                 30.0                  25         0.0       8412 

#### 불균형 데이터 해결 ####
from imblearn.over_sampling import SMOTE

X_notcat = train_notcat.drop('passorfail', axis=1)
y_notcat = train_notcat['passorfail']


# print(X_notcat.isnull().sum())


for column in X_notcat.columns:
    X_notcat[column] = X_notcat[column].fillna(X_notcat[column].mode()[0])
# X_notcat = X_notcat.fillna(X_notcat.mode()[0])

y_notcat = y_notcat.fillna(y_notcat.mode()[0])

print(X_notcat.isnull().sum())

print(X_notcat.shape) #(92013, 20)
print(y_notcat.shape) #(92013,)



sm = SMOTE(random_state=42)
X_res_notcat, y_res_notcat = sm.fit_resample(X_notcat, y_notcat)

X_res_notcat = X_res_notcat.to_numpy()
y_res_notcat = y_res_notcat.to_numpy()

print(X_res_notcat.shape) #(175998, 19)
print(y_res_notcat.shape) #(175998,)

print(type(X_res_notcat))
print(type(y_res_notcat))


X_cat = train_cat.drop('passorfail', axis=1)
y_cat = train_cat['passorfail']


for column in X_cat.columns:
    X_cat[column] = X_cat[column].fillna(X_cat[column].mode()[0])
# X_notcat = X_notcat.fillna(X_notcat.mode()[0])

y_cat = y_cat.fillna(y_cat.mode()[0])


# print(X_cat.isnull().sum())

print(X_cat.shape) #(92013, 20)
print(y_cat.shape) #(92013,)


sm = SMOTE(random_state=42)
X_res_cat, y_res_cat = sm.fit_resample(X_cat, y_cat)

# print(X_cat.shape)
# print(X_res_cat.shape)

print(X_res_cat.columns)

# Index(['count', 'molten_temp', 'facility_operation_cycleTime',
#        'production_cycletime', 'low_section_speed', 'high_section_speed',
#        'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1',
#        'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1',
#        'lower_mold_temp2', 'lower_mold_temp3', 'sleeve_temperature',
#        'physical_strength', 'Coolant_temperature', 'EMS_operation_time',
#        'mold_code'],

print(X_cat.columns)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(X_res_notcat, y_res_notcat, test_size=0.25, random_state=7534, stratify=y_res_notcat)

print(x_train.shape) # (131998, 19)
print(y_train.shape) #(131998,)
print(x_test.shape) #(44000, 19)

x_train = x_train.reshape(131998, 19, 1)
x_test = x_test.reshape(44000, 19, 1)
y_train = y_train.reshape(131998, 1)
y_test = y_test.reshape(44000, 1)

#### 모델 #### 
model = Sequential()
model.add(LSTM(32, return_sequences=False, activation= 'relu', input_shape=(19, 1)))  #행 무시 열 우선 즉 7을 빼준다. #Rnn은 Dense와 바로 연결이 가능하다.  
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    verbose = 1,
    restore_best_weights=True
)

import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))


path = 'C:/Users/ddong40/ai_2/_save/대회/Kaist'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath
)

import time

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                 validation_split= 0.2, callbacks=[es,mcp])
end = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

print(y_predict.shape)
print(y_test.shape)

print(y_predict)
print(y_test)


acc = accuracy_score(y_predict, y_test)

print(acc)

# LSTM, dnn 모델
# acc : 0.9306590909090909

