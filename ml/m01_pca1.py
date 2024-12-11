from sklearn.datasets import load_iris
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

for i in range(len(x[0])):
    x_pca = x.copy()
    pca = PCA(n_components=(i+1))
    x_pca = pca.fit_transform(x_pca)
    x_train, x_test, y_train, y_test = train_test_split(x,  y, train_size=0.8, random_state=10, shuffle=True,
                                                    stratify=y)  
    model = RandomForestClassifier(random_state=124)
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(x.shape)
    print((i+1), '인 경우 model.score :', results)


# #4. 평가

# #model.evaluate가 없다

# results = model.score(x_test, y_test)
# print(x.shape)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=2)
# x = pca.fit_transform(x)

# print(x)
# print(x.shape)

#  #y의 라벨의 개수에 맞춰서 train test의 비율을 정한다.



# #3. 훈련

# model.fit(x_train, y_train)


#4. 평가

#model.evaluate가 없다

 #accuracy와 같다.  #regresser는 r2score를 뽑아준다.

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
