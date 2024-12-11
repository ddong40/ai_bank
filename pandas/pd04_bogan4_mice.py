import numpy as np
import pandas as pd



data = pd.DataFrame([[2, np.nan, 6, 8, 10 ],
                    [2, 4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan, 4, np.nan, 8, np.nan]])
print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)


#pip install impyute


np.float = float
from impyute.imputation.cs import mice

data9 = mice(data.values, n=10, seed=777)

print(data9)
