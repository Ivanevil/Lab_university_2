
# coding: utf-8

# In[4]:


import numpy as np


N[0:9,1]=np.diff(N[:,0])
N[0:8,2]=np.diff(N[0:9,1])
N[0:7,3]=np.diff(N[0:8,2])
N[0:6,4]=np.diff(N[0:7,3])
N[0:5,5]=np.diff(N[0:6,4])
N[0:4,6]=np.diff(N[0:5,5])
N[0:3,7]=np.diff(N[0:4,6])
N[0:2,8]=np.diff(N[0:3,7])
N[0:1,9]=np.diff(N[0:2,8])
np.set_printoptions(suppress = True, precision=5, linewidth=100)
print(N)


# In[6]:


import math as mt
import numpy as np
import matplotlib.pyplot as plt
h = 0.1
x = np.zeros((10))
x = np.arange(7, 7.9, 0.1)


def q(k):
    return ((k - x[0]) / h)


def q2(k, n):
    p = 1
    for i in range(n):
        if i == 0:
            p = p * q(k)
        else:
            p = p * (q(k) - i + 1) / i
    return p


def f(k, n):
    f = N[0, 0]
    for i in range(1, n + 1):
        f += N[0, i] * q2(k, i)
    return f


n = 2
y = []
for i in np.arange(7, 7.9, 0.1):
    y.append(f(i, n))
plt.plot(x, y)
plt.grid()
plt.show()


# In[7]:


import math as mt
import numpy as np
import matplotlib.pyplot as plt
h = 0.1
x = np.zeros((10))
x = np.arange(7, 7.9, 0.1)


def q(k):
    return ((k - x[9]) / h)


def q2(k, n):
    p = 1
    for i in range(n):
        if i == 0:
            p = p * q(k)
        else:
            p = p * (q(k) + i - 1) / i
    return p


def f(k, n):
    f = N[9, 0]
    for i in range(1, n + 1):
        f += N[9 - i, i] * q2(k, i)
    return f


n = 2
y = []
for i in np.arange(7, 7.95, 0.1):
    y.append(f(i, n))
plt.plot(x, y)
plt.grid()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
# tex = '\\frac{1}{\\sqrt{2\\sqrt{2\\pi}}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)'
# a = '\\frac{a}{b}'  #notice escaped slash
# tex = '\\Sigma_{i=0}^{n}\\y_{i}\\frac{\\(x-x_{0})(x-x_{1})\ldots(x-\\x_{i-1})(x-\\x_{i+1})\ldots(x-\\x_{n})}{\\(\\x_{i}-\\x_{0})(\\x_{i}-\\x_{1})\ldots(\\x_{i}-\\x_{i-1})(\\x_{i}-\\x_{i+1})\ldots(\\x_{i}-\\x_{n})}\\right)'
# plt.text(1, 1,'$%s$'%tex, horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=30, color='black')
# plt.plot()
plt.show()


def poz(c, xi, n):
    p = 1
    for i in range(n):
        if(x[i] != xi):
            p = p * (c - x[i]) / (xi - x[i])
    return p


def lagr(c, n):
    s = 0
    for i in range(n):
        s += N[i][0] * poz(c, x[i], n)
    return s


n = 9
y = []
for i in np.arange(7, 7.9, 0.1):
    y.append(lagr(i, n))
print(y)
plt.plot(x, y)


y2 = []
for i in np.arange(7.05, 8, 0.1):
    y2.append(lagr(i, n))
    if (i != 7.05):
        print((i - 0.1), ' : ', y2[int((i - 7.05) * 10)])
        plt.scatter((i - 0.1), y2[int((i - 7.05) * 10)], c='blue')

plt.grid()
plt.show()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
# tex = '\\frac{1}{\\sqrt{2\\sqrt{2\\pi}}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)'
# a = '\\frac{a}{b}'  #notice escaped slash
# tex = '\\Sigma_{i=0}^{n}\\y_{i}\\frac{\\(x-x_{0})(x-x_{1})\ldots(x-\\x_{i-1})(x-\\x_{i+1})\ldots(x-\\x_{n})}{\\(\\x_{i}-\\x_{0})(\\x_{i}-\\x_{1})\ldots(\\x_{i}-\\x_{i-1})(\\x_{i}-\\x_{i+1})\ldots(\\x_{i}-\\x_{n})}\\right)'
# plt.text(1, 1,'$%s$'%tex, horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=30, color='black')
# plt.plot()
plt.show()


def poz(c, xi, n):
    p = 1
    for i in range(n):
        if(x[i] != xi):
            p = p * (c - x[i]) / (xi - x[i])
    return p


def lagr(c, n):
    s = 0
    for i in range(n):
        s += N[i][0] * poz(c, x[i], n)
    return s


n = 9
y = []
for i in np.arange(7, 7.9, 0.1):
    y.append(lagr(i, n))
print(y)
plt.plot(x, y)


y2 = []
for i in np.arange(7.05, 8, 0.1):
    y2.append(lagr(i, n))
    if (i != 7.05):
        print((i - 0.1), ' : ', y2[int((i - 7.05) * 10)])
        plt.scatter((i - 0.1), y2[int((i - 7.05) * 10)], c='blue')

plt.grid()
plt.show()
plt.show()


import numpy as np


def c(k):
    s = 0
    for i in range(n + 1):
        s += x[i]**k
    return s


def b(k):
    s = 0
    for i in range(n + 1):
        s += (x[i]**k) * N[i][0]
    return float(s)


k = input('n: ')
n = int(k)
M2 = np.zeros((n, n))
V2 = np.zeros((n))
for i in range(n):
    for j in range(n):
        M2[i][j] = (c(i + j))
    V2[i] = b(i)
a = np.linalg.solve(M2, V2)


def f(x):
    f = 0
    for i in range(n):
        f += a[i] * (x**i)
    return f


y = []
x = []
for i in np.arange(7, 7.95, 0.1):
    x.append(i)
    y.append(f(i))
print(y)
plt.plot(x, y)

plt.grid()
plt.show()

