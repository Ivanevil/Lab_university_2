
# coding: utf-8

# In[ ]:


import math


def f(x):
    return 5**x - 6 * x - 3


def fx(x):
    return math.log2(5) * 5**x - 6


def fxx(x):
    return ((math.log2(5))**2) * 5**x


def xn(b):
    return b - f(b) / fx(b)


a = 1
b = 2
eps = 0.0001
c = (a + b) / 2

if (fx(c) * fxx(c) > 0):
    x1 = xn(b)
    b = x1
    x2 = xn(b)
    while(abs(x2 - x1) / x1 > eps):
        x1 = x2
        b = x2
        x2 = xn(b)
    print(x1)
else:
    x1 = xn(a)
    a = x1
    x2 = xn(a)
    while(abs(x2 - x1) / x1 > eps):
        x1 = x2
        a = x2
        x2 = xn(a)
    print(x1)

