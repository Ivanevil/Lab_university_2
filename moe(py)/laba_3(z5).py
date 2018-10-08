
# coding: utf-8

# In[25]:


# Интерполяция Полиномом

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

fid = open("./data_var07.txt", "r")
lines = fid.readlines()
fid.close()

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

