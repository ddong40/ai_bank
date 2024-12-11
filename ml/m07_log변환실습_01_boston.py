from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

datasets = load_boston()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot
# df.plot.box()
# plt.show()

# print(df.info()) #결측치 없고~
# print(df.describe())

# df['B'].plot.box() #시리즈에서 이거 됨
# plt.show()

# df['B'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

x['B'] = np.log1p(x['B']) #지수변환 np.exp1m  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

y_train = np.log1p(y_train)
y_test = np.log1p(y_train)

model = LinearRegression()