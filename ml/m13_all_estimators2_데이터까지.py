#애는 리그레서로 맹그러

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings 
import time
warnings.filterwarnings('ignore')

boston = load_boston(return_X_y=True)
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)


datasets = [boston, california, diabetes]
data_name = ['보스턴', '캘리포니아', '디아베트']


#1. 데이터
# x, y = load_boston(return_X_y=True)

#2. 모델구성
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

print('all Algorithms : ', all)

print('모델의 개수', len(all)) #모델의 개수 41
# 모델의 개수 55

print('sk 버전 :', sk.__version__) #sk 버전 : 1.1.3

#튜플의 0번째거 빼서 돌리라! 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

start_time = time.time()
for index, value in enumerate(datasets) : #enuerate를 사용하면 인덱스 값까지 함께 반환해준다.
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=123, train_size=0.8, 
        # stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  
    
    maxName = ''
    maxAccuracy = 0  

    for name, model in all:
        try:
            #2. 모델
            model = model()
            #3. 훈련
            scores  = cross_val_score(model, x_train, y_train, cv=kfold)
            x = round(np.mean(scores), 4)
            # print( '=============={}, {}============='.format(data_name[index],name))
            # print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))
            y_predict = cross_val_predict(model, x_test, y_test)
            acc = r2_score(y_test, y_predict)
            #4. 평가
            # print('cross_val_predict acc : ', acc)
            
            if x > maxAccuracy:
                maxAccuracy = x
                maxName = name 
            
            
        except:
            pass
            # print(name, '은 바보 멍충이!!!')

    print('===============', data_name[index], '==================')
    print('최고모델 : ', maxName)
    print('acc : ', maxAccuracy, '\n')    
        
end_time = time.time()

print('걸린시간 : ', round(end_time - start_time, 2), '초')
