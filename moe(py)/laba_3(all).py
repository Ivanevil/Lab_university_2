
# coding: utf-8

# In[1]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


fid = open("./data_var07.txt", "r")
lines = fid.readlines()
fid.close()


# In[3]:


# 1
l1 = [float(lines[i].split(',')[1]) for i in range(1, len(lines))]
l2 = [float(lines[i].split(',')[2]) for i in range(1, len(lines))]
l3 = [float(lines[i].split(',')[3]) for i in range(1, len(lines))]
l4 = [float(lines[i].split(',')[4]) for i in range(1, len(lines))]
l5 = [float(lines[i].split(',')[5]) for i in range(1, len(lines))]

fig = plt.figure()
graph1 = plt.plot(l1)
plt.grid(True)

fig2 = plt.figure()
graph2 = plt.plot(l2)
plt.grid(True)

fig3 = plt.figure()
graph3 = plt.plot(l3)
plt.grid(True)

fig4 = plt.figure()
graph4 = plt.plot(l4)
plt.grid(True)

fig5 = plt.figure()
graph5 = plt.plot(l5)
plt.grid(True)

plt.show()


# In[4]:


# 2
l1_new = l1[0:10]

x = [i for i in range(10)]
x_new = [i / 10 for i in range(100)]


def lang(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * (p1 / p2)
    return z


y_new = [lang(x, l1_new, i) for i in x_new]
plt.plot(x_new, y_new)
plt.plot(l1_new, "o")
plt.ylim(26, 30)
plt.grid(True)
plt.show()


# In[5]:


# 3
l1_new = l1[0:5]

x = [i for i in range(5)]
x_new = [i / 10 for i in range(100)]

C = []


def product(val, n):
    mul = 1
    for i in range(n):
        if i:
            mul *= val - x[i - 1]
        yield mul


for n in range(len(x)):
    p = product(x[n], n + 1)
    C.append((l1_new[n] - sum(C[k] * next(p) for k in range(n))) / next(p))


def f(v):
    return sum(C[k] * p for k, p in enumerate(product(v, len(C))))


y_new = [f(i) for i in x_new]
plt.plot(x_new, y_new)
plt.plot(l1_new, "o")
plt.ylim(15, 40)
plt.grid(True)
plt.show()


# In[6]:


# 4
l1_new = l1[0:5]

x = [i for i in range(5)]
x_new = [i / 10 for i in range(100)]

C = []


def product(val, n):
    mul = 1
    for i in range(n):
        if i:
            mul *= val + x[i - 1]   # изменение знака на +
        yield mul


for n in range(len(x)):
    p = product(x[n], n + 1)
    C.append((l1_new[n] - sum(C[k] * next(p) for k in range(n))) / next(p))


def f(v):
    return sum(C[k] * p for k, p in enumerate(product(v, len(C))))


y_new = [f(i) for i in x_new]
plt.plot(x_new, y_new)
plt.plot(l1_new, "o")
plt.ylim(15, 40)
plt.grid(True)
plt.show()


# In[7]:


# 5
y = [float(lines[i].split(',')[1]) for i in range(1, len(lines))]
x = [i for i in range(len(y))]

d = 2  # степень полинома
fp = sp.polyfit(x, y, d)
f = sp.poly1d(fp)

y1 = [fp[0] * x[i]**2 + fp[1] * x[i] + fp[2] for i in range(0, len(x))]  # значения функции a*x**2+b*x+c


fx = sp.linspace(x[0], x[-1] + 1, len(x))

plt.plot(y)
plt.plot(y1)
plt.grid(True)
plt.show()
