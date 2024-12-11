import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

path = 'c:/Users/ddong40/Desktop/본프로젝트/AI Trainer/'

df = pd.read_csv(path + "플랭크_전체_1104.csv", encoding='cp949')

x = df.drop('description', axis=1).values
y = df['description']

# 다중 클래스 인코딩
#le = LabelEncoder()
#y = le.fit_transform(y)

# x 값 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 스케일링된 x 값을 데이터프레임으로 변환
x_scaled_df = pd.DataFrame(x_scaled, columns=df.drop('description', axis=1).columns)

scaled_df = pd.concat([x_scaled_df, y], axis = 1)

# 스케일링된 x 값을 CSV로 저장
scaled_df.to_csv(path + "플랭크_scaled_values.csv", index=False, encoding='utf-8-sig')

print("저장완료")
