from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

#1 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

df = pd.DataFrame(x, columns=datasets.feature_names)

df['Target'] = y

import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib 
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,
            cbar=True)
plt.show()