import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = "C:/Users/ddong40/ai_2/_data/kaggle/wine_quality/"

train = pd.read_csv(path + 'train.csv', index_col=0)

x = train.drop(['quality'], axis=1)
y = train['quality']

data = y.groupby(train['quality']).count()
print(data)
# quality
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5

plt.bar(data.index, data.values)
plt.show()