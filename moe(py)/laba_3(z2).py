
# coding: utf-8

# In[11]:


# Интерполяция Лагранжем

import numpy as np
import matplotlib.pyplot as plt

fid = open("./data_var07.txt", "r")
lines = fid.readlines()
fid.close()

l1 = [float(lines[i].split(',')[1]) for i in range(1, len(lines))]
l1_new = l1[0:10]

x = [i for i in range(10)]
x_new = [i / 10 for i in range(150)]


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
