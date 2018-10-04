
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt

fid = open("./data_var07.txt", "r")
lines = fid.readlines()
fid.close()

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

