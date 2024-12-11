import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 로지스틱리그레션은 분류
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

#1 데이터

x, y = load_breast_cancer(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=4444, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state = 4444,
                        #   bootstrap=True, # 디폴트, 중복허용, 모델이 겹쳐지는 경우가 있음
                          bootstrap=False) 
# model = LogisticRegression()
# model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종 점수 : ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score', acc)


# 최종 점수 :  0.9473684210526315
# acc_score 0.9473684210526315 

#bagging 방식은 모델을 여러개 생성한 뒤 n빵하는 것이다. 그 과정이 복잡하여 사이킷런에서 제공해준다. 

# bagging
# 최종 점수 :  0.9385964912280702
# acc_score 0.9385964912280702

# logisticRegression
# 최종 점수 :  0.9736842105263158
# acc_score 0.9736842105263158

# DecisionTree Bagging, bootstraip false
# 최종 점수 :  0.9649122807017544
# acc_score 0.9649122807017544

# logistic Bagging, 