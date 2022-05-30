import numpy as np
import matplotlib.pylab as plt

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

x = np.arange(-5, 5, 0.1)
y = softmax(x)
plt.plot(x, y)
plt.show()
