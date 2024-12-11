from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer, load_wine, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_digits()
x = dataset.data
y = dataset.target
print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    stratify=y
                                                    )


#2. 모델구성
model = RandomForestClassifier(random_state=Random_state)

model.fit(x_train, y_train)
print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('acc', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

percentiles = np.percentile(model.feature_importances_, 25)

indexs = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= percentiles:
        indexs.append(index)

x_train = np.delete(x_train, indexs, axis = 1)
x_test = np.delete(x_test, indexs, axis = 1)
model.fit(x_train, y_train)

print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
print('acc', model.score(x_test, y_test))
print(np.percentile(model.feature_importances_, 25))
print(model.feature_importances_)

# col = []
# for index, importance in enumerate(model.feature_importances_):
#     if importance < 0.01:
#         col.append(index)

# print(col)

# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9583333333333334
# 0.002210329533326792
# [0.00000000e+00 2.24223392e-03 2.20030420e-02 1.13709365e-02
#  9.54252544e-03 2.36638585e-02 6.84456454e-03 9.57013092e-04
#  3.44700479e-05 1.27065807e-02 2.99843194e-02 6.80007878e-03
#  1.61911101e-02 2.42502696e-02 5.72047895e-03 4.08041734e-04
#  1.29020372e-05 1.08343412e-02 1.84309091e-02 2.58302091e-02
#  3.03274343e-02 4.83097762e-02 7.96831595e-03 4.92011361e-04
#  1.47714458e-04 1.22436966e-02 4.18854551e-02 2.36037708e-02
#  3.79908459e-02 2.09645449e-02 3.44052259e-02 0.00000000e+00
#  0.00000000e+00 3.43155310e-02 2.48951553e-02 1.64828484e-02
#  3.41923138e-02 1.74473201e-02 2.93964859e-02 0.00000000e+00
#  0.00000000e+00 9.21952972e-03 4.12545058e-02 4.05221809e-02
#  2.11435997e-02 1.96680433e-02 2.33173059e-02 2.74530687e-05
#  3.48887100e-05 2.25190947e-03 1.83911777e-02 1.87886687e-02
#  1.21854320e-02 2.25616277e-02 1.91757031e-02 2.11461637e-03
#  3.43871859e-05 1.97249686e-03 2.06432240e-02 1.12249370e-02
#  2.43538972e-02 3.06602286e-02 1.37990114e-02 3.75884488e-03]
# ======================== RandomForestClassifier Random_state =  8888 ======================
# acc 0.9611111111111111
# 0.01218147117615748
# [0.00201647 0.02348945 0.01037343 0.01017038 0.02193907 0.00785059
#  0.01319056 0.02568739 0.0068711  0.01518798 0.02709572 0.00537721
#  0.00809735 0.01844881 0.02563652 0.03008397 0.05446276 0.01058269
#  0.01516356 0.04211926 0.0254556  0.02923395 0.02588777 0.03104079
#  0.02971257 0.02675136 0.01549991 0.04446543 0.01754437 0.02656032
#  0.00965695 0.03594249 0.04283691 0.01732505 0.01713319 0.01703794
#  0.00266676 0.01611958 0.02317616 0.0127144  0.01972043 0.02439854
#  0.02501991 0.00923403 0.02573121 0.0319163  0.01963907 0.00373474]