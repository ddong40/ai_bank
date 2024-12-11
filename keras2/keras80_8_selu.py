import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

selu = lambda x: np.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (np.exp(x) - 1))

y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()