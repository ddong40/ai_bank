import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


#1. 데이터
x, y = load_iris(return_X_y=True)

print(x)

n_splits = 5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

#2 모델
model = SVC()

#3. 훈련

scores  = cross_val_score(model, x, y, cv=kfold) #cv -> cross validation
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4)) #ACC :  [1.         0.86666667 1.         0.96666667 0.96666667]  #평균 ACC :  0.96