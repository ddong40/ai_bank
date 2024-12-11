from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5, 5, 1))) #2,2 는 몇으로 쪼개줄지, input_shape에서 5,5,1 가로 5 세로 5에 컬러는 흑백이다.
model.add(Conv2D(5, (2,2))) # (3, 3, 5)

model.summary()
# 여기서 10은 아웃풋 
# 압축한 그림을 클론으로 증폭한다. 
# 4,4 로 압축한 데이터를 레이어를 통과시키면 10개가 된다.
# conglution을 너무 많이하면 데이터가 소실 된다.  