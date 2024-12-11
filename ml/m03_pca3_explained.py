from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

#RandomForestClassfier는 분류
#RandomForestRegressor는 회귀

#1 데이터

datasets = load_iris()
x = datasets['data']
y = datasets.target 
print(x.shape, y.shape) #(150, 4) (150,)






x_train, x_test, y_train, y_test = train_test_split(x,  y, train_size=0.8, random_state=10, shuffle=True,
                                                    stratify=y) #y의 라벨의 개수에 맞춰서 train test의 비율을 정한다.

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=4)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x)
print(x.shape)

model = RandomForestClassifier(random_state=124)

#3. 훈련

model.fit(x_train, y_train)


#4. 평가

#model.evaluate가 없다

results = model.score(x_test, y_test)
print(x_test.shape)
print(x_train.shape)
print('model.score :', results) #accuracy와 같다.  #regresser는 r2score를 뽑아준다.

#pca 없을 때 
# (150, 4)
# model.score : 0.9333333333333333

# pca 살려
# (150, 3)
# model.score : 0.9

# pca components가 2개인 것과 4개와 결과가 같다.
# (150, 2)
# model.score : 0.9333333333333333

# (150, 1)
# model.score : 0.9333333333333333



# train_split 뒤에 했을 경우

# (30, 1)
# (120, 1)
# model.score : 0.9666666666666667

# (30, 2)
# (120, 2)
# model.score : 1.0

# (30, 3)
# (120, 3)
# model.score : 1.0

# (30, 4)
# (120, 4)
# model.score : 1.0

evr = pca.explained_variance_ratio_ #설명가능한 변화율
print(evr) #[0.74200394 0.21452087 0.03872095] 선을 그을 때 마다 생성되는 변화율
print(sum(evr)) #0.995245754627282

evr_cumsum = np.cumsum(evr)
print(evr_cumsum) #[0.74200394 0.95652481 0.99524575 1.        ]

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()