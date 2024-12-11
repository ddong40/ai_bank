
param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4)}

def y_function(x1, x2):
    return -x1 **2 - (x2-2) **2 +10

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds=param_bounds,
    
) # 내가 만들어낼 함수의 최소 값을 찾아내는 것이다. y function의 인수의 값 x1, x2 를 반환한다. 
# -1에서 5사이 0에서 4사이의 가장 좋은 놈을 찾아내는 것이다. 
# p  bounds는 y를 최소값을 만들기 위한 x1과  x2의 최소 값을 찾으라는 뜻
# 만일 int형일 경우 35번을 돌 것이다. 

optimizer.maximize(init_points=5,
                   n_iter=20)

print(optimizer.max)
