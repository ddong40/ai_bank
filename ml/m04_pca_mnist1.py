from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x =np.concatenate([x_train, x_test], axis=0)


print(x.shape) #(70000, 28, 28)

x = np.reshape(x, (70000, 28*28))

pca = PCA(n_components=784)
x = pca.fit_transform(x)


cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 1.0) + 1 #index라서 하나가 더 들어가 줘야한다. 
print('선택할 차원 수 :', d)


print(np.argmax(cumsum >= 0.95) + 1) #154
print(np.argmax(cumsum >= 0.99) + 1) #331
print(np.argmax(cumsum >= 0.999) + 1) #486
print(np.argmax(cumsum >= 1.0) + 1) #713

#pca는 evr로 확인하고 수치에 대하여 


############## [실습] ##################
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 이상 #154개
# 0.99 이상 #331개
# 0.999 이상 #486개
# 1.0 일때 몇개? #713개

# argmax와 csum을 이용하여