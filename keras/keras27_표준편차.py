import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1,2,3,1],
                [4,5,6,2],
                [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]])

#1. 평균

means = np.mean(data, axis=0)
print('평균 : ', means) #평균 :  [ 7.  8.  9. 47.]


#2. 모집단 분산 (제곱하여 더한뒤 n빵)
population_variances = np.var(data, axis=0, ) #ddof=0 디폴트 
print('분산 : ',population_variances) 
# 분산 :  [  18.   18.   18. 3038.]

#3. 표본 분산 (n-1빵) #값이 더 커진다. 나눠주는 수가 작기 때문에 
variances = np.var(data, axis=0, ddof=1) # ddof=1은 n-1하겠다는 뜻
print('표본 분산 : ', variances)
# 표본 분산 :  [  22.5   22.5   22.5 3797.5]

#4. 표준편차
std = np.std(data, axis=0, ddof=1)
print('표준편차 : ', std)
# 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5. StandardScaler 
scaler = StandardScaler()

scaler_data = scaler.fit_transform(data)

print('StandardScaler : ', scaler_data)

# 편차 [-6, -3, 0, 3, 6]
# 분산 (36 9 0 9 36) /4 
# 표본 분산 22.5
# 모집단 분산 18
# 계산시 모집단 분산으로 StandardScaler가 적용된 것을 확인할 수 있다. 

