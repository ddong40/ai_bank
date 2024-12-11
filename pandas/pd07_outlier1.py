import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print('1사분위 : ', quartile_1) #4.0
    print('q2 : ', q2) #7.0
    print('3사분위 : ', quartile_3) #10.0
    iqr = quartile_3 - quartile_1  #6.0
    print('iqr : ', iqr)
    lower_bound = quartile_1 = (iqr *1.5) # 통상적으로 1qr은 1.5를 곱하는 것이 디폴트이다.
    upper_bound = quartile_3 = (iqr *1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound)), iqr
    
outliers_loc, iqr = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.axhline(iqr, color='red', label='IQR')
plt.show()