import numpy as np
from sklearn.datasets import load_boston, load_iris
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
        acc = r2_score(y_test, y_predict)
        #4. 평가
        print('cross_val_predict r2 : ', acc)
        
    except:
        print(name, '은 바보 멍충이!!!')
        



# ==============ARDRegression=============
# ACC :  [0.90651431 0.93961652 0.95765966 0.94176099 0.90081328]
# 평균 ACC :  0.9293
# cross_val_predict r2 :  0.9185791346411343
# ==============AdaBoostRegressor=============
# ACC :  [0.85591077 0.93984962 0.996928   0.8875     0.89719591]
# 평균 ACC :  0.9155
# cross_val_predict r2 :  0.95
# ==============BaggingRegressor=============
# ACC :  [0.93440514 0.9512782  0.99384615 0.9665     0.88501393]
# 평균 ACC :  0.9462
# cross_val_predict r2 :  0.9075
# ==============BayesianRidge=============
# ACC :  [0.90513553 0.93754123 0.95827024 0.94153835 0.90180027]
# 평균 ACC :  0.9289
# cross_val_predict r2 :  0.9218318391423134
# CCA 은 바보 멍충이!!!
# ==============DecisionTreeRegressor=============
# ACC :  [0.84565916 0.93984962 1.         0.95       0.86629526]
# 평균 ACC :  0.9204
# cross_val_predict r2 :  0.95
# ==============DummyRegressor=============
# ACC :  [-0.00502412 -0.03524436 -0.0400641   0.         -0.00435237]
# 평균 ACC :  -0.0169
# cross_val_predict r2 :  -0.028124999999999734
# ==============ElasticNet=============
# ACC :  [0.48127216 0.42243528 0.45262091 0.40135457 0.46949538]
# 평균 ACC :  0.4454
# cross_val_predict r2 :  0.418366053312232
# ==============ElasticNetCV=============
# ACC :  [0.90605974 0.93835643 0.95776017 0.94159416 0.90142831]
# 평균 ACC :  0.929
# cross_val_predict r2 :  0.9059585667636131
# ==============ExtraTreeRegressor=============
# ACC :  [0.92282958 0.93984962 1.         0.9        0.86629526]
# 평균 ACC :  0.9258
# cross_val_predict r2 :  1.0
# ==============ExtraTreesRegressor=============
# ACC :  [0.91532862 0.93166316 0.99686154 0.964785   0.90192758]
# 평균 ACC :  0.9421
# cross_val_predict r2 :  0.96547
# GammaRegressor 은 바보 멍충이!!!
# ==============GaussianProcessRegressor=============
# ACC :  [0.56098358 0.24013096 0.62019889 0.48264097 0.87577221]
# 평균 ACC :  0.5559
# cross_val_predict r2 :  0.48407259526680924
# ==============GradientBoostingRegressor=============
# ACC :  [0.94864524 0.92821602 0.98821803 0.92762385 0.89994408] 
# 평균 ACC :  0.9385
# cross_val_predict r2 :  0.9211642816183054
# ==============HistGradientBoostingRegressor=============
# ACC :  [0.91547201 0.93518439 0.99422904 0.96069784 0.95462535]
# 평균 ACC :  0.952
# cross_val_predict r2 :  -0.028124999999999734
# ==============HuberRegressor=============
# ACC :  [0.90280123 0.93747773 0.95753426 0.93522293 0.9046128 ]
# 평균 ACC :  0.9275
# cross_val_predict r2 :  0.898812062795636
# IsotonicRegression 은 바보 멍충이!!!
# ==============KNeighborsRegressor=============
# ACC :  [0.89813505 0.89172932 0.9808547  0.968      0.92779944]
# 평균 ACC :  0.9333
# cross_val_predict r2 :  0.856
# ==============KernelRidge=============
# ACC :  [-0.90435412 -0.65910036 -0.82464231 -0.44850245 -0.68414626]
# 평균 ACC :  -0.7041
# cross_val_predict r2 :  -0.8442623982076971
# ==============Lars=============
# ACC :  [0.90663605 0.93727582 0.95795613 0.94102894 0.90025916]
# 평균 ACC :  0.9286
# cross_val_predict r2 :  0.9124579098638321
# ==============LarsCV=============
# ACC :  [0.90719992 0.93865207 0.95798532 0.94087003 0.90025916]
# 평균 ACC :  0.929
# cross_val_predict r2 :  0.9051685449852526
# ==============Lasso=============
# ACC :  [-0.00502412 -0.03524436 -0.0400641   0.         -0.00435237]
# 평균 ACC :  -0.0169
# cross_val_predict r2 :  -0.028124999999999734
# ==============LassoCV=============
# ACC :  [0.90717865 0.93868312 0.95807939 0.94089245 0.90115006] 
# 평균 ACC :  0.9292
# cross_val_predict r2 :  0.9059412434607459
# ==============LassoLars=============
# ACC :  [-0.00502412 -0.03524436 -0.0400641   0.         -0.00435237]
# 평균 ACC :  -0.0169
# cross_val_predict r2 :  -0.028124999999999734
# ==============LassoLarsCV=============
# ACC :  [0.90719992 0.93865207 0.95798532 0.94087003 0.90025916]
# 평균 ACC :  0.929
# cross_val_predict r2 :  0.9048943887176657
# ==============LassoLarsIC=============
# ACC :  [0.90772734 0.93880427 0.95836334 0.94136808 0.90025916]
# 평균 ACC :  0.9293
# cross_val_predict r2 :  0.9074793398699221
# ==============LinearRegression=============
# ACC :  [0.90663605 0.93727582 0.95795613 0.94102894 0.90025916]
# 평균 ACC :  0.9286
# cross_val_predict r2 :  0.9124579098638331
# ==============LinearSVR=============
# ACC :  [0.89571421 0.93792017 0.95810352 0.93745705 0.90499308]
# 평균 ACC :  0.9268
# cross_val_predict r2 :  0.9127583985401466
# ==============MLPRegressor=============
# ACC :  [0.93966121 0.91768867 0.97754963 0.9232393  0.91577709]
# 평균 ACC :  0.9348
# cross_val_predict r2 :  0.9258108438229972
# MultiOutputRegressor 은 바보 멍충이!!!
# MultiTaskElasticNet 은 바보 멍충이!!!
# MultiTaskElasticNetCV 은 바보 멍충이!!!
# MultiTaskLasso 은 바보 멍충이!!!
# MultiTaskLassoCV 은 바보 멍충이!!!
# ==============NuSVR=============
# ACC :  [0.90640025 0.94526834 0.98526241 0.93095463 0.92124582]
# 평균 ACC :  0.9378
# cross_val_predict r2 :  0.8764668025104195
# ==============OrthogonalMatchingPursuit=============
# ACC :  [0.87074234 0.920389   0.92780376 0.91391911 0.8825615 ]
# 평균 ACC :  0.9031
# cross_val_predict r2 :  0.9129289603107987
# ==============OrthogonalMatchingPursuitCV=============
# ACC :  [0.90663605 0.93727582 0.95795613 0.9227324  0.90025916]
# 평균 ACC :  0.925
# cross_val_predict r2 :  0.9009056705688681
# PLSCanonical 은 바보 멍충이!!!
# ==============PLSRegression=============
# ACC :  [0.89470826 0.92955439 0.95479892 0.94398113 0.90234192]
# 평균 ACC :  0.9251
# cross_val_predict r2 :  0.8851443721838177
# ==============PassiveAggressiveRegressor=============
# ACC :  [0.70517791 0.89563303 0.91122625 0.77835077 0.89720078]
# 평균 ACC :  0.8375
# cross_val_predict r2 :  0.7727255275549763
# ==============PoissonRegressor=============
# ACC :  [0.66747239 0.69019688 0.6975423  0.68631093 0.70443455] 
# 평균 ACC :  0.6892
# cross_val_predict r2 :  0.751064861953506
# ==============QuantileRegressor=============
# ACC :  [-3.21543232e-03 -2.25563888e-02 -2.56410240e-02  4.15629087e-10
#  -2.78551456e-03]
# 평균 ACC :  -0.0108
# cross_val_predict r2 :  8.254052996647943e-11
# ==============RANSACRegressor=============
# ACC :  [0.91255073 0.93682419 0.95795613 0.94147587 0.90025916]
# 평균 ACC :  0.9298
# cross_val_predict r2 :  0.9089945161937559
# ==============RadiusNeighborsRegressor=============
# ACC :  [ 7.97265759e-01 -2.77394648e+17  9.71361136e-01 -4.61168602e+17
#   9.22812518e-01]
# 평균 ACC :  -1.477126501027758e+17
# cross_val_predict r2 :  -6.917529036231017e+17
# ==============RandomForestRegressor=============
# ACC :  [0.91346881 0.94446917 0.99837949 0.963575   0.93330808]
# 평균 ACC :  0.9506
# cross_val_predict r2 :  0.92937
# RegressorChain 은 바보 멍충이!!!
# ==============Ridge=============
# ACC :  [0.90393034 0.93767421 0.95833789 0.94170421 0.90356125]
# 평균 ACC :  0.929
# cross_val_predict r2 :  0.9236756978749056
# ==============RidgeCV=============
# ACC :  [0.90393034 0.93767421 0.95833789 0.94170421 0.90356125]
# 평균 ACC :  0.929
# cross_val_predict r2 :  0.9115461646032721
# ==============SGDRegressor=============
# ACC :  [0.86911538 0.92729148 0.95253383 0.92734654 0.91706407]
# 평균 ACC :  0.9187
# cross_val_predict r2 :  0.8317653847602471
# ==============SVR=============
# ACC :  [0.89298474 0.94349425 0.97715614 0.93014124 0.922898  ]
# 평균 ACC :  0.9333
# cross_val_predict r2 :  0.8783434735642629
# StackingRegressor 은 바보 멍충이!!!
# ==============TheilSenRegressor=============
# ACC :  [0.90286543 0.9278843  0.95384558 0.92776809 0.90920069]
# 평균 ACC :  0.9243
# cross_val_predict r2 :  0.9183600566691542
# ==============TransformedTargetRegressor=============
# ACC :  [0.90663605 0.93727582 0.95795613 0.94102894 0.90025916]
# 평균 ACC :  0.9286
# cross_val_predict r2 :  0.9124579098638331
# ==============TweedieRegressor=============
# ACC :  [0.78841836 0.8296675  0.88906704 0.86188029 0.86905598]
# 평균 ACC :  0.8476
# cross_val_predict r2 :  0.796050628557874
# VotingRegressor 은 바보 멍충이!!!