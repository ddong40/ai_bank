
import numpy as np
from sklearn.datasets import load_boston, load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb
import time

# iris
# 06_cancer
# 09_wine
# 11_digits




#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

# x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=123, train_size=0.8, 
#     stratify=y
# )

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

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
        stratify=y
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
            acc = accuracy_score(y_test, y_predict)
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

# print('max name : ', maxName, 'max accurcay :', maxAccuracy)

# =============== 아이리스 ==================
# 최고모델 :  LinearDiscriminantAnalysis
# acc :  0.975

# =============== 캔서 ==================
# 최고모델 :  LogisticRegression
# acc :  0.9824

# =============== 와인 ==================
# 최고모델 :  LinearDiscriminantAnalysis
# acc :  1.0

# =============== 디지트 ==================
# 최고모델 :  SVC
# acc :  0.9777

# 걸린시간 :  91.39 초