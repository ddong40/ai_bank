#애는 리그레서로 맹그러

import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings 
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    #stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# all = all_estimators(type_filter='classifier')
all = all_estimators(type_filter='regressor')

print('all Algorithms : ', all)

print('모델의 개수', len(all)) #모델의 개수 41
# 모델의 개수 55

print('sk 버전 :', sk.__version__) #sk 버전 : 1.1.3

#튜플의 0번째거 빼서 돌리라! 

maxName = ''
maxAccuracy = 0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

for name, model in all:
    try:
        #2. 모델
        model = model()
        #3. 훈련
        scores  = cross_val_score(model, x_train, y_train, cv=kfold)
        print( '=============={}============='.format(name))
        print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test)
        acc = accuracy_score(y_test, y_predict)
        #4. 평가
        print('cross_val_predict r2 : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')
        
# ==============ARDRegression=============
# ACC :  [0.63868074 0.80326571 0.64950009 0.75565513 0.80326603]
# 평균 ACC :  0.7301
# ARDRegression 은 바보 멍충이!!!
# ==============AdaBoostRegressor=============
# ACC :  [0.76349816 0.88627498 0.78268633 0.80934433 0.79310178]
# 평균 ACC :  0.807
# AdaBoostRegressor 은 바보 멍충이!!!
# ==============BaggingRegressor=============
# ACC :  [0.87233345 0.84410649 0.80187891 0.85745672 0.88757232]
# 평균 ACC :  0.8527
# BaggingRegressor 은 바보 멍충이!!!
# ==============BayesianRidge=============
# ACC :  [0.66326568 0.80011305 0.64899875 0.73525937 0.80721508]
# 평균 ACC :  0.731
# BayesianRidge 은 바보 멍충이!!!
# CCA 은 바보 멍충이!!!
# ==============DecisionTreeRegressor=============
# ACC :  [0.53362616 0.57705045 0.76118199 0.59724473 0.79484863]
# 평균 ACC :  0.6528
# DecisionTreeRegressor 은 바보 멍충이!!!
# ==============DummyRegressor=============
# ACC :  [-0.02042588 -0.00380497 -0.0151419  -0.00379291 -0.00050612]
# 평균 ACC :  -0.0087
# DummyRegressor 은 바보 멍충이!!!
# ==============ElasticNet=============
# ACC :  [0.61378521 0.69820252 0.5481866  0.76382153 0.67612589]
# 평균 ACC :  0.66
# ElasticNet 은 바보 멍충이!!!
# ==============ElasticNetCV=============
# ACC :  [0.65967005 0.80116452 0.64714    0.73506048 0.79838304] 
# 평균 ACC :  0.7283
# ElasticNetCV 은 바보 멍충이!!!
# ==============ExtraTreeRegressor=============
# ACC :  [0.74376188 0.74532961 0.63991876 0.620712   0.75436   ]
# 평균 ACC :  0.7008
# ExtraTreeRegressor 은 바보 멍충이!!!
# ==============ExtraTreesRegressor=============
# ACC :  [0.92088772 0.93943931 0.81059447 0.87949684 0.88858727]
# 평균 ACC :  0.8878
# ExtraTreesRegressor 은 바보 멍충이!!!
# ==============GammaRegressor=============
# ACC :  [0.67999553 0.66333593 0.54781249 0.75488686 0.67312792]
# 평균 ACC :  0.6638
# GammaRegressor 은 바보 멍충이!!!
# ==============GaussianProcessRegressor=============
# ACC :  [-0.10011669  0.18563898  0.14629962  0.2571788   0.68805235]
# 평균 ACC :  0.2354
# GaussianProcessRegressor 은 바보 멍충이!!!
# ==============GradientBoostingRegressor=============
# ACC :  [0.89363156 0.88847247 0.85272696 0.87831398 0.88689774]
# 평균 ACC :  0.88
# GradientBoostingRegressor 은 바보 멍충이!!!
# ==============HistGradientBoostingRegressor=============
# ACC :  [0.85734264 0.90232283 0.78284771 0.89321474 0.87547845]
# 평균 ACC :  0.8622
# HistGradientBoostingRegressor 은 바보 멍충이!!!
# ==============HuberRegressor=============
# ACC :  [0.67365455 0.78706976 0.62123757 0.75271374 0.77700355]
# 평균 ACC :  0.7223
# HuberRegressor 은 바보 멍충이!!!
# IsotonicRegression 은 바보 멍충이!!!
# ==============KNeighborsRegressor=============
# ACC :  [0.77078147 0.78876115 0.58994399 0.74260466 0.8468651 ]
# 평균 ACC :  0.7478
# KNeighborsRegressor 은 바보 멍충이!!!
# ==============KernelRidge=============
# ACC :  [ -5.6576963   -4.70130817  -4.86509472 -10.52011856  -4.90005219]
# 평균 ACC :  -6.1289
# KernelRidge 은 바보 멍충이!!!
# ==============Lars=============
# ACC :  [0.66230011 0.80378441 0.65206483 0.52031949 0.78899082]
# 평균 ACC :  0.6855
# Lars 은 바보 멍충이!!!
# ==============LarsCV=============
# ACC :  [0.66230011 0.80406569 0.64929488 0.52449093 0.78943134]
# 평균 ACC :  0.6859
# LarsCV 은 바보 멍충이!!!
# ==============Lasso=============
# ACC :  [0.59985393 0.75257776 0.59355763 0.78887393 0.70549672]
# 평균 ACC :  0.6881
# Lasso 은 바보 멍충이!!!
# ==============LassoCV=============
# ACC :  [0.65983166 0.80121149 0.65021793 0.72159501 0.80200915]
# 평균 ACC :  0.727
# LassoCV 은 바보 멍충이!!!
# ==============LassoLars=============
# ACC :  [-0.02042588 -0.00380497 -0.0151419  -0.00379291 -0.00050612]
# 평균 ACC :  -0.0087
# LassoLars 은 바보 멍충이!!!
# ==============LassoLarsCV=============
# ACC :  [0.66230011 0.80147561 0.64929488 0.71901253 0.80257471]
# 평균 ACC :  0.7269
# LassoLarsCV 은 바보 멍충이!!!
# ==============LassoLarsIC=============
# ACC :  [0.66230011 0.80072565 0.64881422 0.71662892 0.8085957 ]
# 평균 ACC :  0.7274
# LassoLarsIC 은 바보 멍충이!!!
# ==============LinearRegression=============
# ACC :  [0.66230011 0.80082316 0.65206483 0.71662892 0.81009207]
# 평균 ACC :  0.7284
# LinearRegression 은 바보 멍충이!!!
# ==============LinearSVR=============
# ACC :  [0.6704925  0.78313383 0.60108876 0.75366337 0.76477764]
# 평균 ACC :  0.7146
# LinearSVR 은 바보 멍충이!!!
# ==============MLPRegressor=============
# ACC :  [0.65183893 0.64900466 0.57960195 0.61429922 0.74148798]
# 평균 ACC :  0.6472
# MLPRegressor 은 바보 멍충이!!!
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# ==============NuSVR=============
# ACC :  [0.55582678 0.63630926 0.51789577 0.72089584 0.64504048]
# 평균 ACC :  0.6152
# NuSVR 은 바보 멍충이!!!
# ==============OrthogonalMatchingPursuit=============
# ACC :  [0.5116866  0.62656178 0.46828951 0.57073488 0.57802883]
# 평균 ACC :  0.5511
# OrthogonalMatchingPursuit 은 바보 멍충이!!!
# ==============OrthogonalMatchingPursuitCV=============
# ACC :  [0.5898069  0.80025884 0.62552306 0.7782056  0.7785408 ]
# 평균 ACC :  0.7145
# OrthogonalMatchingPursuitCV 은 바보 멍충이!!!
# PLSCanonical 은 바보 멍충이!!!
# ==============PLSRegression=============
# ACC :  [0.65166215 0.78598865 0.60848064 0.77088667 0.76807897]
# 평균 ACC :  0.717
# PLSRegression 은 바보 멍충이!!!
# ==============PassiveAggressiveRegressor=============
# ACC :  [0.26218441 0.71926701 0.54578626 0.61381525 0.54586455]
# 평균 ACC :  0.5374
# PassiveAggressiveRegressor 은 바보 멍충이!!!
# ==============PoissonRegressor=============
# ACC :  [0.7661719  0.84488207 0.68690624 0.84053227 0.8464397 ]
# 평균 ACC :  0.797
# PoissonRegressor 은 바보 멍충이!!!
# ==============QuantileRegressor=============
# ACC :  [-4.45818161e-05 -5.10642774e-03 -4.55669433e-02 -4.83802048e-02
#  -2.34813743e-02]
# 평균 ACC :  -0.0245
# QuantileRegressor 은 바보 멍충이!!!
# ==============RANSACRegressor=============
# ACC :  [ 0.46389585  0.66520574 -0.18869538  0.17798546  0.18029054]
# 평균 ACC :  0.2597
# RANSACRegressor 은 바보 멍충이!!!
# ==============RadiusNeighborsRegressor=============
# ACC :  [nan nan nan nan nan]
# 평균 ACC :  nan
# RadiusNeighborsRegressor 은 바보 멍충이!!!
# ==============RandomForestRegressor=============
# ACC :  [0.84302693 0.87292756 0.78958063 0.88299321 0.87714223]
# 평균 ACC :  0.8531
# RandomForestRegressor 은 바보 멍충이!!!
# RegressorChain 은 바보 멍충이!!!
# ==============Ridge=============
# ACC :  [0.66247992 0.80079859 0.651481   0.7207272  0.80972373]
# 평균 ACC :  0.729
# Ridge 은 바보 멍충이!!!
# ==============RidgeCV=============
# ACC :  [0.66419007 0.79909662 0.64594369 0.7207272  0.80522799]
# 평균 ACC :  0.727
# RidgeCV 은 바보 멍충이!!!
# ==============SGDRegressor=============
# ACC :  [0.65424974 0.79657453 0.64717569 0.73506493 0.80962873]
# 평균 ACC :  0.7285
# SGDRegressor 은 바보 멍충이!!!
# ==============SVR=============
# ACC :  [0.56825021 0.67660629 0.52924157 0.74515562 0.67486823]
# 평균 ACC :  0.6388
# SVR 은 바보 멍충이!!!
# StackingRegressor 은 바보 멍충이!!!
# ==============TheilSenRegressor=============
# ACC :  [0.5263886  0.63180508 0.4707283  0.61700676 0.6981633 ]
# 평균 ACC :  0.5888
# TheilSenRegressor 은 바보 멍충이!!!
# ==============TransformedTargetRegressor=============
# ACC :  [0.66230011 0.80082316 0.65206483 0.71662892 0.81009207]
# 평균 ACC :  0.7284
# TransformedTargetRegressor 은 바보 멍충이!!!
# ==============TweedieRegressor=============
# ACC :  [0.63627285 0.66491381 0.52444062 0.73619758 0.66673812]
# 평균 ACC :  0.6457
# TweedieRegressor 은 바보 멍충이!!!
# VotingRegressor 은 바보 멍충이!!!        