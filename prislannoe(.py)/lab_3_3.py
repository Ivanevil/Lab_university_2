
# coding: utf-8

# In[ ]:


import math as mt
import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):
    return mt.sin(x - y) - x * y + 1


def f2(x, y):
    return x**2 - y**2 - 0.75


def f1x(x, y):
    return mt.cos(x - y) - y


def f2x(x, y):
    return 2 * x


def f1y(x, y):
    return -mt.cos(x - y) - x


def f2y(x, y):
    return -2 * y


def y(x):
    return mt.sqrt(abs(x**2 - 0.75))


x = []
p = []
for i in np.arange(0, 3, 0.5):
    p.append(f1(i, y(i)))
    x.append(i)
plt.plot(x, p)
plt.grid()
plt.show()

import math as mt
x1 = 1.3
y1 = y(x1)
eps = 0.0001
g = (f1(x1, y1) * f2y(x1, y1) - f2(x1, y1) * f1y(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
h = (f1x(x1, y1) * f2(x1, y1) - f2x(x1, y1) * f1(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
x2 = x1 - g
y2 = y1 - h
while (mt.sqrt((x2 - x1)**2 + (y2 - y1)**2) > eps):
    x1 = x2
    y1 = y2
    g = (f1(x1, y1) * f2y(x1, y1) - f2(x1, y1) * f1y(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
    h = (f1x(x1, y1) * f2(x1, y1) - f2x(x1, y1) * f1(x1, y1)) / (f1x(x1, y1) * f2y(x1, y1) - f1y(x1, y1) * f2x(x1, y1))
    x2 = x1 - g
    y2 = y1 - h
print(x1, y1)

