
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import math


def f1(x):
    return 5**x - math.e**(-x) - 1


y1 = []
y2 = []
x = range(-1, 3)
for i in range(-1, 3):
    y1.append(f1(i))
plt.plot(x, y1)
plt.grid(True)
plt.show()

x1 = 1
eps = 0.0001
gamm = 0.05
x2 = x1 - gamm * f1(x1)
while (abs(x2 - x1) > eps):
    x1 = x2
    x2 = x1 - gamm * f1(x1)
print(x1)

