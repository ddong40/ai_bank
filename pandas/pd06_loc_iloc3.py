## loc는 location의 약자
## index로 될 경우 loc사용
## iloc는 int location 
## iloc는 int인 숫자일 경우 
import warnings
import pandas as pd
print(pd.__version__) # 2.2.2
# warnings.ignore

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500']
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)

#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print('======================================================================')

print('시가가 1100원 이상인 행을 모두 출력')

aaa = df['시가'] >= '1100'
print(aaa)

print(df[aaa])

#      종목명    시가    종가
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000

print(df.loc[aaa])

# print(df.iloc[aaa]) #에러

### 1100원 이상인 놈만 뽑아서 새로 df2를 만들거야.
### 이 방식을 여러분은 제일 많이 쓸거야!!! ### 




df2 = df[df['시가'] >= '1100']
print(df2)

print('=======================')

# df2 = df.loc[aaa] # 똑같음

df3 = df[df['시가'] >= '1100']['종가']
print(df3)

df4 = df.loc[df['시가'] >= '1100']['종가']
print(df4)

df6 = df.loc[df['시가'] >= '1100', '종가']
print(df6)