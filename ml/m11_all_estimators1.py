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
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y
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

for name, model in all:
    try:
        #2. 모델
        model = model()
        #3. 훈련
        model.fit(x_train, y_train)
        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '은 바보 멍충이!!!')

# ARDRegression 의 정답률 :  0.8967860021515403
# AdaBoostRegressor 의 정답률 :  0.9419321241283778
# BaggingRegressor 의 정답률 :  0.925
# BayesianRidge 의 정답률 :  0.8994707924912021
# CCA 은 바보 멍충이!!!
# DecisionTreeRegressor 의 정답률 :  0.75
# DummyRegressor 의 정답률 :  0.0
# ElasticNet 의 정답률 :  0.46342182410124777
# ElasticNetCV 의 정답률 :  0.8967818669688222
# ExtraTreeRegressor 의 정답률 :  0.8
# ExtraTreesRegressor 의 정답률 :  0.92692
# GammaRegressor 은 바보 멍충이!!!
# GaussianProcessRegressor 의 정답률 :  0.7800988873219343
# GradientBoostingRegressor 의 정답률 :  0.9272084040433899
# HistGradientBoostingRegressor 의 정답률 :  0.909740355823849
# HuberRegressor 의 정답률 :  0.8993429470408626
# IsotonicRegression 은 바보 멍충이!!!
# KNeighborsRegressor 의 정답률 :  0.92
# KernelRidge 의 정답률 :  -0.5343351209920235
# Lars 의 정답률 :  0.9008782085074352
# LarsCV 의 정답률 :  0.8971557063280101
# Lasso 의 정답률 :  0.0
# LassoCV 의 정답률 :  0.8970323124308976
# LassoLars 의 정답률 :  0.0
# LassoLarsCV 의 정답률 :  0.8971557063280101
# LassoLarsIC 의 정답률 :  0.8967747761067266
# LinearRegression 의 정답률 :  0.9008782085074353
# LinearSVR 의 정답률 :  0.9004947685317449
# MLPRegressor 의 정답률 :  0.8909870002520006
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# NuSVR 의 정답률 :  0.9268463528378843
# OrthogonalMatchingPursuit 의 정답률 :  0.915047606826263
# OrthogonalMatchingPursuitCV 의 정답률 :  0.9008782085074353
# PLSCanonical 은 바보 멍충이!!!
# PLSRegression 의 정답률 :  0.8960658801820887
# PassiveAggressiveRegressor 의 정답률 :  -0.2996075080146139
# PoissonRegressor 의 정답률 :  0.682212201744762
# QuantileRegressor 의 정답률 :  4.2922432275105393e-10
# RANSACRegressor 의 정답률 :  0.8955000839120191
# RadiusNeighborsRegressor 의 정답률 :  0.8875138856237881
# RandomForestRegressor 의 정답률 :  0.92913
# RegressorChain 은 바보 멍충이!!!
# Ridge 의 정답률 :  0.898256287886421
# RidgeCV 의 정답률 :  0.8982562878864412
# SGDRegressor 의 정답률 :  0.8653759246834036
# SVR 의 정답률 :  0.9282598013920788
# StackingRegressor 은 바보 멍충이!!!
# TheilSenRegressor 의 정답률 :  0.8929167089370529
# TransformedTargetRegressor 의 정답률 :  0.9008782085074353
# TweedieRegressor 의 정답률 :  0.8249493142524955
# VotingRegressor 은 바보 멍충이!!!