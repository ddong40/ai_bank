from sklearn.datasets import load_diabetes
import warnings 
warnings.filterwarnings('ignore')



# print(x.shape, y.shape) #(442, 10) (442,)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score, r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) #442, 10



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

#kfold
parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], #0.2, 0.3], 
    'max_depth' : [3, 4, 5,6,8]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0]},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0]}]


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)



model = HalvingRandomSearchCV(xgb.XGBRegressor(
    tree_method = 'hist',
    device = 'cuda:0',
    n_estimators=50
                                              ), 
                    parameters, cv=kfold,
                    verbose=1,
                    refit=True,
                    #  n_iter = 9,
                    random_state= 333,
                    factor=2,
                    # min_resources=50,
                    aggressive_elimination=False
                     ) 

start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = False)

end_time = time.time()



print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_leaf': 10, 'max_depth': 12, 'learning_rate': 0.01}
# 최적의 파라미터 :  {'learning_rate': 0.05, 'subsample': 0.7}
# 최적의 파라미터 :  {'learning_rate': 0.05, 'colsample_bytree': 0.7}

print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.2861891833948846
# 최고의 점수 :  0.29140564463573365

# gridsearch
# 최고의 점수 :  0.3965239637685497

# random
# 최고의 점수 :  0.3745725199525287

print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.3141311535904062
# 모델의 점수 :  0.27295663927518

# gridsearch
# 모델의 점수 :  0.4258449944542835

# randomsearch
# 모델의 점수 :  0.385456959816478
y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 
# accuracy_score :  0.3141311535904062
# accuracy_score :  0.27295663927518

# gridsearch
# accuracy_score :  0.4258449944542835

# randomsearch
# accuracy_score :  0.385456959816478


y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.3141311535904062
# 최적 튠 ACC:  0.27295663927518

# gridsearch
# 최적 튠 ACC:  0.4258449944542835

# randomsearch
# 최적 튠 ACC:  0.385456959816478
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  149.31 초
# 걸린시간 :  61.75 초
# 걸린시간 :  235.7 초