import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

#1. 데이터
np.random.seed(777)
x = 2* np.random.rand(100, 1) -1 #-1부터 1까지의 난수 생성
print(x)
print(np.max(x), np.min(x))
print(len(x))

exit()

y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1) # y = 3x^2 2x + 1 + 노이즈 
# randn은 평균0, 표준편차 1 정규 분포를 따르는 난수 생성
# rand은 0과 1사이의 랜덤한 수로 100x1 배열 생성

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2. 모델

model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)

# 원본(x)

plt.scatter(x, y, color='blue', label = 'Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
# plt.show()

# 다항식 회귀 그래프 그리기
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_poly = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot2 = model2.predict(x_test_poly)
plt.plot(x_test, y_plot, color='red', label='기냥')
plt.plot(x_test, y_plot2, color='green', label='Polynomial Regression')
plt.legend()
plt.show()