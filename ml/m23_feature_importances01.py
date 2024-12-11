from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#1 데이터

x, y = load_iris(return_X_y=True)

print(x.shape, y.shape)
# (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123, stratify=y)



#2 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

print('random_state : ', random_state)

for model in models:
    model.fit(x_train, y_train)
    print("===================", model.__class__.__name__, "====================")
    print('acc', model.score(x_test, y_test)) 
    print(model.feature_importances_) #각각의 피쳐가 얼마나 공헌했는지에 대한 명세를 제공한다. 


# =================== DecisionTreeClassifier(random_state=777) ====================
# acc 0.8888888888888888
# [0.00857143 0.03428571 0.41033009 0.54681277]
# =================== RandomForestClassifier(random_state=777) ====================
# acc 0.9555555555555556
# [0.09859761 0.03106655 0.44597491 0.42436093]
# =================== GradientBoostingClassifier(random_state=777) ====================
# acc 0.9555555555555556
# [0.00569573 0.02500574 0.65183202 0.31746651]
# =================== XGBClassifier ====================
# acc 0.9555555555555556
# [0.03295855 0.02776272 0.75007254 0.18920612]

