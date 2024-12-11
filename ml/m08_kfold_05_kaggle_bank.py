# https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv
# 1-3열 index처리
# 문자를 수치화 해주기 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time


path = 'C:/Users/ddong40/ai_2/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

print(train_csv.shape) 
print(test_csv.shape)

encoder = LabelEncoder()



test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])

x = train_csv.drop(['CustomerId','Surname','Exited'], axis=1)
y = train_csv['Exited']

print(x.shape) #(165034, 10)
print(test_csv.shape)  #(110023, 10)


from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import xgboost as xgb


n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2 모델
model = xgb.XGBClassifier()

#3. 훈련

scores  = cross_val_score(model, x, y, cv=kfold) #cv -> cross validation
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

#kfold
# ACC :  [0.86469537 0.86296846 0.86651316 0.86190808 0.86420651] 
# 평균 ACC :  0.8641

#staryfiedkfold
# ACC :  [0.86311994 0.8670888  0.86387736 0.86402884 0.86117676] 
# 평균 ACC :  0.8639


exit()
from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# x = pca.fit_transform(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print( np.argmax(cumsum>=1.0)+1) #1
# print( np.argmax(cumsum>=0.999)+1) #2
# print( np.argmax(cumsum>=0.99)+1) #2
# print( np.argmax(cumsum>=0.95)+1) #2



# from sklearn.preprocessing import MinMaxScaler
# scalar=MinMaxScaler()
# x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle= True, random_state= 512)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

list_a = np.array([1,2,10])

for i in reversed(list_a):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    

    # print(x_train.shape)
    # print(x_test.shape)

    # print(x_test)
    # print(test_csv)


    #2 모델구성
    model = Sequential()
    model.add(Dense(64,activation='relu', input_dim=i))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3 컴파일 훈련
    from sklearn.metrics import accuracy_score


    model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    start_time = time.time()
    es = EarlyStopping(
        monitor= 'val_loss',
        mode = 'min',
        patience= 30,
        restore_best_weights= True    
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')

    path1 = './_save/keras30_mcp/08_kaggle_bank/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path1, 'k30_', date, '_', filename])

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        patience = 30,
        verbose = 0,
        save_best_only=True,
        filepath = filepath
    )


    model.fit(x_train, y_train, epochs= 1000, batch_size=100, verbose=0, validation_split= 0.25, callbacks=[es, mcp])
    end_time = time.time()

    #4 평가 예측,
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)
    print("로스 : ", loss[0])
    print("acc : ", round(loss[1], 3))

    y_pred = np.round(y_pred)

    # accuracy_score(y_test, y_pred)

    print("acc스코어 : ", accuracy_score)
    print("걸린시간 : ", round(end_time - start_time, 2), "초" )
    print('-------------------------------------------------')

    y_submit = model.predict(test_csv)
    # y_submit = np.round(y_submit)
    y_submit_binary = np.round(y_submit).astype(int)

    #5 파일 생성
    sampleSubmission['Exited'] = y_submit_binary

    sampleSubmission.to_csv(path+'samplesubmission_0723_1239.csv')

    print(sampleSubmission['Exited'].value_counts())
    
# 로스 :  0.21177111566066742
# acc :  0.788
# acc스코어 :  <function accuracy_score at 0x0000028F98123EE0>
# 걸린시간 :  29.34 초

# binary 
# 로스 :  0.5086240172386169
# acc :  0.788
# acc스코어 :  <function accuracy_score at 0x000001FBDAD03F70>
# 걸린시간 :  26.37 초

#그냥 정규화하기
# 로스 :  0.10367565602064133
# acc :  0.857
# acc스코어 :  <function accuracy_score at 0x000002ADB5832F70>
# 걸린시간 :  46.48 초
# 0    109698
# 1       325

# balance 삭제 안하고 정규화 하기
# 로스 :  0.09938615560531616
# acc :  0.864
# acc스코어 :  <function accuracy_score at 0x0000023DFDA84EE0>
# 걸린시간 :  48.49 초
# 0    59900
# 1    50123

# 로스 :  0.09936358034610748
# acc :  0.865
# acc스코어 :  <function accuracy_score at 0x000002C44E054EE0>
# 걸린시간 :  292.57 초
# 0    90285
# 1    19738

# minmaxscaler
# 로스 :  0.3239017426967621
# acc :  0.863

# standardscaler
# 로스 :  0.32506412267684937
# acc :  0.864

# maxabsscaler
# 로스 :  0.3243679106235504
# acc :  0.863

# RobustScaler
# 로스 :  0.3252757489681244
# acc :  0.865

# 세이브 값
# 로스 :  0.32415467500686646
# acc :  0.864