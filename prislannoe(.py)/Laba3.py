
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):
    return x**3 + y**3 - 6


def f2(x, y):
    return y - (math.e)**(-x)


def f1x(x, y):
    return 3 * x**2


def f2x(x, y):
    return math.log2(math.e) * math.e**(-x)


def f1y(x, y):
    return 3 * y**2


def f2y(x, y):
    return 1


def y(x):
    return (math.e)**(-x)


x = []
p = []
for i in np.arange(0, 3, 0.5):
    p.append(f1(i, y(i)))
    x.append(i)
plt.plot(x, p)
plt.grid()
plt.show()


x1 = 1.7
y1 = y(x1)
eps = 0.0001
g = (f1(x1, y1) * f2y(x1, y1) - f2(x1, y1) * f1y(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
h = (f1x(x1, y1) * f2(x1, y1) - f2x(x1, y1) * f1(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
x2 = x1 - g
y2 = y1 - h
while (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) > eps):
    x1 = x2
    y1 = y2
    g = (f1(x1, y1) * f2y(x1, y1) - f2(x1, y1) * f1y(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
    h = (f1x(x1, y1) * f2(x1, y1) - f2x(x1, y1) * f1(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
    x2 = x1 - g
    y2 = y1 - h
print(x1, y1)

