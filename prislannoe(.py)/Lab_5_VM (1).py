
# coding: utf-8

# In[2]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math as mt


# In[8]:


e = 0.001
n = 100
def f(t, p):
    s = np.sin(t-5)/(1+p*t+p*(t**2))
    return s
def integral(i, p):
    s = 0
    l = 5/i
    anach = np.arange(0, 5, l)
    bkonh = np.arange(l, 5+l, l)
    s += ((bkonh-anach)/90)*(7*f(anach, p) + 32*f((3*anach+bkonh)/4, p) + 12*f((anach+bkonh)/2, p) + 32*f((anach+3*bkonh)/4, p)+7*f(bkonh, p))
#     print(s)
    return s
N = np.zeros((n))
y = []
x = []
for k in range (n):
    i=1
    a=1
    b=0
    w=0
    Flag = False
#     print(integral(i, k)/integral(i*2, k)-1)
    while Flag==False :
        b = a
        a = np.sum(integral(i, k))
#         print(a, b, abs(a/b-1))
        if (abs(a/b-1) < e):
            Flag = True
        
        if k==1:
            y.append(np.sum(integral(i, k)))
            x.append(i)
            w= w+1
        i=i*2 
#         print(i)
    N[k]=np.sum(integral(i, k)) - 10
x1 = np.zeros((n))
y1 = np.zeros((n))
for i in range (n):
    x1[i] = i
    y1[i]=f(i, 1)
#     print(N[k])
plt.plot(x, y)
plt.grid()
plt.show()
plt.plot(x1, y1)
plt.grid()
plt.show()


# In[ ]:




