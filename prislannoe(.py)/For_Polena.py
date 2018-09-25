
# coding: utf-8

# In[3]:


import scipy as sp
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import pylab
from matplotlib import mlab


def f(t, p):
    n = np.sin(t) * np.exp(- p * t)
    return n


def simpson(a, b, p):
    fla = ((b - a) / 8) * (f(a, p) + 3 * f((2 * a + b) / 3, p) + 3 * f((a + 2 * b) / 3, p) + f(b, p))
    return (fla)


def rims(i):
    mas = []
    it1 = simpson(0, 1, i)
    mas.append(it1)
    it2 = simpson(0, 0.5, i) + simpson(0.5, 1, i)
    mas.append(it2)
    j = 2
    while (abs(it1 / it2 - 1) > 0.01):
        j += 1
        it1 = it2
        it2 = 0
        for k in range(j):
            it2 += simpson(k / j, (k + 1) / j, i)
        mas.append(it2)
    #print (j)
    return (mas)

    # return (it2)


mas = rims(100)
print(mas)
print(len(mas))
dlina = len(mas)

int = []
int.append(0)
i = 1
while i < 100:
    mas = rims(i)
    int.append(mas[len(mas) - 1])
    i += 1


tlist = mlab.frange(0, 1, 0.001)
ylist = [f(t, 100) for t in tlist]
plt.xlabel('x')
plt.ylabel('y')
plt.title('График подынтегральной функции')
pylab.plot(tlist, ylist)
plt.grid(True)
pylab.show()


xlist = mlab.frange(1, dlina, 1)
ylist = mas
#plt.axis([-10, 10, -10,10])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Значение интеграла от номера итерации')
pylab.plot(xlist, ylist)
plt.grid(True)
pylab.show()
