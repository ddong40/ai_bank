#리더보드 테스트용

학생csv = 'jena_전사영.csv'

path1 = 'C://ai_2//_data//kaggle//jena//'
path2 = 'C://ai_2//_save//keras55//'

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)

print(datasets)
print(datasets.shape) 

y_정답 = datasets.iloc[-144: ,1]
print(y_정답)
print(y_정답.shape) # (144,)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

print(y_정답[:5])
print(학생꺼[:5])

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_정답, 학생꺼)
print('RMSE :', rmse)