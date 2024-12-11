# 06_cancer
# 09_wine
# 11_digits

### 요 파일에 이 3개의 데이터셋 다 넣어서 23번처럼 맹그러 ###

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


#1 데이터

x, y = load_breast_cancer(return_X_y=True)
x1, y1 = load_wine(return_X_y=True)
x2, y2 = load_digits(return_X_y=True)

print(x.shape, y.shape)
# (150, 4) (150,)
print(x1.shape, y1.shape)
# (178, 13) (178,)
print(x2.shape, y2.shape)
# (1797, 64) (1797,)


random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123, stratify=y)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, shuffle=True, random_state=123, stratify=y1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, shuffle=True, random_state=123, stratify=y2)

data = [[x_train, x_test, y_train, y_test], [x1_train, x1_test, y1_train, y1_test], [x2_train, x2_test, y2_train, y2_test]]
data_name = []


print(data[0])



#2 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

print('random_state : ', random_state)

for model in models:
    for i in data:
        model.fit(i[0], i[2])
        print("===================", model.__class__.__name__, "====================")
        print('acc', model.score(i[1], i[3])) 
        print(model.feature_importances_) #각각의 피쳐가 얼마나 공헌했는지에 대한 명세를 제공한다. 
