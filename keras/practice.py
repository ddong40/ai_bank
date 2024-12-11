from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

#RandomForestClassfier는 분류
#RandomForestRegressor는 회귀

#1 데이터

datasets = load_iris()
x = datasets['data']
y = datasets.target 
print(x.shape, y.shape) #(150, 4) (150,)

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)


# for i in range(len(x[0])):
#     x_pca = x.copy()
#     pca = PCA(n_components=(i+1))
#     x_pca = pca.fit_transform(x_pca)
#     x_train, x_test, y_train, y_test = train_test_split(x,  y, train_size=0.8, random_state=10, shuffle=True,
#                                                     stratify=y)  
#     model = RandomForestClassifier(random_state=124)
#     model.fit(x_train, y_train)
#     results = model.score(x_test, y_test)
#     print(x.shape)
#     print((i+1), '인 경우 model.score :', results)