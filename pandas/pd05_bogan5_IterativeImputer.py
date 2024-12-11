import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

data = pd.DataFrame([[2, np.nan, 6, 8, 10 ],
                    [2, 4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan, 4, np.nan, 8, np.nan]])
print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

imputer = IterativeImputer() #BayesianRidge 회귀 모델.
data1 = imputer.fit_transform(data)
print(data1)

imputer = IterativeImputer(estimator=DecisionTreeRegressor()) #BayesianRidge 회귀 모델.
data2 = imputer.fit_transform(data)
print(data2)

# [[ 2.  2.  2.  4.]
#  [ 6.  4.  4.  4.]
#  [ 6.  4.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

imputer = IterativeImputer(estimator=DecisionTreeRegressor()) #BayesianRidge 회귀 모델.
data3 = imputer.fit_transform(data)
print(data3)

# [[ 2.  2.  2.  4.]
#  [ 6.  4.  4.  4.]
#  [ 6.  4.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

imputer = IterativeImputer(estimator=XGBRegressor()) #BayesianRidge 회귀 모델.
data4 = imputer.fit_transform(data)
print(data4)

# [[ 2.          2.          2.          4.00096321]
#  [ 2.00112057  4.          4.          4.        ]
#  [ 6.          4.00000906  6.          4.00096321]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.99906492 10.          7.99903679]]