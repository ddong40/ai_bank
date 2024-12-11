import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10 ],
                    [2, 4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan, 4, np.nan, 8, np.nan]])
print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

# 0.  결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 1. 결측치 삭제
# print(data.dropna()) #디폴트 true 있는 행 모두 삭제

# print(data.dropna(axis=0)) #true 있는 행 삭제
print(data.dropna(axis=1)) #true 있는 열 삭제

#2-1. 특정값 - 평균
means = data.mean()
print(means)

data2 = data.fillna(means) #평균을 채워넣음 기준은 열 기준으로 채워 넣는다. #행단위로 평균을 내면 다른 열의 값들과의 평균을 내게 되기 때문에 적합하지 않음
print(data2)

#2-2. 특정값 - 중위값
med = data.median()
print(med)
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
data3 = data.fillna(med)
print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

#2-3. 특정값 - 0 채우기 / 임의값 채우기
data4 = data.fillna(0)
print(data4)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

data4_2 = data.fillna(777)
print(data4_2)
#       x1     x2    x3     x4
# 0    2.0    2.0   2.0  777.0
# 1  777.0    4.0   4.0    4.0
# 2    6.0  777.0   6.0  777.0
# 3    8.0    8.0   8.0    8.0
# 4   10.0  777.0  10.0  777.0

#2-4. 특정값 - ffill (통상 마지막 값, )
data5 = data.ffill()
print(data5)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

data5 = data.fillna(method='ffill')
print(data5) #위의 것과 동일

#2-5. 특정값 - bfill
data6 = data.bfill()
print(data6)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

data6 = data.fillna(method='bfill')
print(data6) #위의 것과 동일

##########################특정 컬럼만 ###############################
means = data['x1'].mean()
print(means) #6.5

meds = data['x4'].median()
print(meds) #6.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(meds)
data['x2'] = data['x2'].ffill()

print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0

