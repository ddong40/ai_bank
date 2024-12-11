import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

leaky_relu = lambda x : np.maximum(0.01*x, x)

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

