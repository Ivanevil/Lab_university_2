
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import math


def f1(x):
    return 5**x - 6 * x - 3


def f2(x):
    return math.log2(5) * 5**x / 6


def fi(x):
    return (5**x - 3) / 6


y1 = []
y2 = []
x = range(-2, 4)
for i in range(-2, 4):
    y1.append(f1(i))
    y2.append(f2(i))
# fig = plt.figure()
plt.plot(x, y1)
plt.grid(True)
plt.show()
plt.plot(x, y2)
plt.grid(True)
plt.show()


def fi(x):
    return (5**x - 3) / 6


x1 = 0
eps = 0.0001
gamm = 0.005
x2 = fi(x1)
while (abs(x2 - x1) > eps):
    x1 = x2
    x2 = fi(x1)
print(x1)


def fi(x):
    return (5**x - 3) / 6


def f1(x):
    return 5**x - 6 * x - 3


x1 = 2
eps = 0.0001
gamm = 0.05
x2 = x1 - gamm * f1(x1)
while (abs(x2 - x1) > eps):
    x1 = x2
    x2 = x1 - gamm * f1(x1)
print(x1)

