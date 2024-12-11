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
# print(df[0]) # error
# print(df['031']) # error
print(df['시가']) # 잘 출력 됨.


#### 아모레 출력하고 싶어
# print(df[3, 0]) #key 에러
print(df['종목명']['045'])

print('======================================================================')

print(df.iloc[3][0])

print(df.loc['045']['종목명'])

print('=================================================================')

print(df.loc['045']['종가'])
print(df.loc['045','종가'])
print(df.loc['045'].loc['종가'])

print(df.iloc[3][2])
print(df.iloc[3, 2])
print(df.iloc[3].iloc[2])

print(df.loc['045'].iloc[2])