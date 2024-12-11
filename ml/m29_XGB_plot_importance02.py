from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #decision tree의 앙상블 형태
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split


#1 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123)


# data = [x_train, x_test, y_train, y_test]
# data_name = ['캘리포니아']

# print(data[0])



#2 모델 구성
# model1 = DecisionTreeRegressor(random_state=random_state)
# model2 = RandomForestRegressor(random_state=random_state)
# model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

model4.fit(x_train, y_train)

# models = [model1, model2, model3, model4]

# print('random_state : ', random_state)

import matplotlib.pyplot as plt
import numpy as np

# for model in models:
#     model.fit(x_train, y_train)
#     print("===================", model.__class__.__name__, "====================")
#     print('acc', model.score(x_test, y_test)) 
#     print(model.feature_importances_) 
    
import matplotlib.pyplot as plt
import numpy as np

from xgboost.plotting import plot_importance
plot_importance(model4) 
plt.show()

# i=0
# for model in models:
#     model.fit(x_train, y_train)
#     print("===================", model.__class__.__name__, "====================")
#     print('acc', model.score(x_test, y_test)) 
#     print(model.feature_importances_)
#     plt.subplot(2,2, i+1)
#     def plot_feature_importances_dataset(model) : 
#         n_features = datasets.data.shape[1]
#         plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#         plt.yticks(np.arange(n_features), datasets.feature_names)
#         plt.xlabel("Feature Importances")
#         plt.ylabel("Features")
#         plt.ylim(-1, n_features)
#         plt.title(model.__class__.__name__)
        
#     plot_feature_importances_dataset(model)
#     i=i+1   
# plt.tight_layout() 
# plt.show()

