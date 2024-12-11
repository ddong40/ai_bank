from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
#1 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target

print(datasets)

df = pd.DataFrame(x, columns=datasets.feature_names)

print(df)

df['Target'] = y
print(df)

print('================================ 상관계수 히트맵 ===============================')
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib 
print(sns.__version__)
print(matplotlib.__version__)
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,
            cbar=True)
plt.show()

