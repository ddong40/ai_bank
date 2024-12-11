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

print(df.iloc[3:, 1])
# 045    3500
# 023     100

print(df.iloc[[2, 4],1]) # LG와 네이버의 시가 / 특정 행만 뽑기

# print(df.iloc[3:, '시가']) # 에러
# print(df.iloc[[2, 4], '시가']) # 에러

print('=========================== loc 아모레와 네이버의 시가 ============================')

print(df.loc['045':'023', '시가']) # 문자열로 들어갈 때 
print(df.loc['045':, '시가'])
print(df.loc[['033', '023'], '시가'])

# 045    3500
# 023     100
# Name: 시가, dtype: object
# 045    3500
# 023     100
# Name: 시가, dtype: object

print(df.loc['033' : '023'].iloc[2])

print(df.iloc[2].loc['시가']) # 2000
