import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2 모델
model = SVC()

#3. 훈련

scores  = cross_val_score(model, x_train, y_train, cv=kfold) #cv -> cross validation #기준 점수 확인 
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4)) 

y_predict = cross_val_predict(model, x_test, y_test) #cv를 안넣어도 돌아가는 이유는 default가 5이다. 

print(y_predict)
print(y_test)
# [1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 1 1 2 2 1 1 1 1 1]
# [1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc) #cross_val_predict ACC : 0.8666666666666667 데이터가 작아서 성능이 떨어짐

