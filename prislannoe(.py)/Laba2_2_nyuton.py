
# coding: utf-8

# In[ ]:


import math


def f(x):
    return 5**x - math.e**(-x) - 1


def fx(x):
    return math.log2(5) * 5**x - math.log2(math.e) * math.e**(-x)


def fxx(x):
    return ((math.log2(5))**2) * 5**x - (math.log2(math.e))**2 * math.e**(-x)


def new(b):
    return b - f(b) / fx(b)


a = 0
b = 1
eps = 0.0001
c = (a + b) / 2

if (fx(c) * fxx(c) > 0):
    x1 = new(b)
    b = x1
    x2 = new(b)
    while(abs(x2 - x1) / x1 > eps):
        x1 = x2
        b = x2
        x2 = new(b)
    print(x1)
else:
    x1 = new(a)
    a = x1
    x2 = new(a)
    while(abs(x2 - x1) / x1 > eps):
        x1 = x2
        a = x2
        x2 = new(a)
    print(x1)

