
# coding: utf-8

# In[2]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import copy


# ## Лабораторная работа №5
# Численное решение системы линейных уравнений, численное интегрирование

# ** Цель работы: **
# научиться считать определенные интеграллы и решать системы линейных уравнений численными методами и с помощью встроенных библиотек. 

# ** Вычисление интеграла :  **
# используем квадратурную формулу Буля(n=4)
# $$ \sum^{4}_{k=0}C^{(4)}_{k}f(x_{k}^{(4)}) = \frac{(\beta - \alpha)}{90}(7f(\alpha) + 32f(\frac{3\alpha +\beta}{4}) + 12f(\frac{\alpha +\beta}{2}+ 32f(\frac{\alpha +3\beta}{4}) + 7f(\beta)) $$

# In[1]:


e = 0.001
n = 10
def f(t, p):
    s = np.sin(t-5)/(1+p*t+p*(t**2))
    return s
def integral(i, p):
    s = 0
    sped = 0
    ot = 10
    while (abs(np.sum(s)-sped)/abs(np.sum(s)))>e:
        l = ot/i
        sped = np.sum(s)
        anach = np.arange(0, ot, l)
        bkonh = np.arange(l, ot+l, l)
        s += ((bkonh-anach)/90)*(7*f(anach, p) + 32*f((3*anach+bkonh)/4, p) + 12*f((anach+bkonh)/2, p) + 32*f((anach+3*bkonh)/4, p)+7*f(bkonh, p))
        ot+=1
        if ot>100:
            print('breake')
            break
        print(np.sum(s))
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
x1 = np.zeros((6))
y1 = np.zeros((6))
for i in range (n):
#     x1[i] = i
#     y1[i]=f(i, 1)
    print(N[k])


# In[ ]:


def f2(t, p):
    s = mt.sin(t)*mt.exp(- p**2 *t**2)
    return s


# Построим графики подинтегральной функции и график значения интеграла от номера итерации (приведены примеры для первой функции)

# In[55]:



plt.plot(x1, y1)
plt.grid()
plt.show()
plt.plot(x, y)
plt.grid()
plt.show()


# In[27]:


s = np.zeros((4))
lo = np.zeros((4))


def integraltrabs(i, p):
    s1 = 0
    l = 5/i
    anach = np.arange(0, 5.01, l)
    lp = np.zeros((i))
    for j in range (i):
        lp[j]=f(anach[j], 1)
    print(lp)
    s1 = np.trapz(lp, dx=(5/(i)))

    return s1
for j in range(1, 4):
    i=2**j
    s[j]=integraltrabs(i, 1)
    lo[j] = i
plt.plot(lo, s)
plt.grid()
plt.show()


# Зададим матрицу, содержащую коэффициенты уравнений:

# In[4]:


def coeff(i):
    return 10+(mt.cos(i)/(i**2+i+1))
n = 100
M=np.zeros((n,n))
M[0,0] = 10
M[n-1,n-1] = 1
M[n-1,n-2] = 1
M[n-2,n-1] = 1
for i in range(1,n-1):
    M[i, i] = coeff(i)
    M[i-1, i] = 1
    M[i, i-1] = 1
np.set_printoptions(suppress=True, precision =4, linewidth=100)
print(M)


# Решим для n=1000 и n=10000 методом Гаусса-Зейделя систему 
# $$x_1 = 10$$
# $$x_{i-1}+(10+cos(i)/(i^2+i+1))x_i+x_{i+1}= -10+\int_0^5\frac{sin(t+5)}{1+it+it^2}dt, i=\overline{2,n-1}$$
# $$x_n = 1$$
# 
# Метод Гаусс-Зейделя
# 
# Итерационными называются приближенные методы, в которых решение системы получается как предел последовательности векторов $$(x^k)_{k=1}^{\infty}$$, каждый последующий элемент которой вычисляется по некоторому единому правилу. Начальный элемент $$x^1$$ выбирается произвольно.
# Условие сходимости
# $${\parallel\frac{x^{k+1}-x^{k}}{x^k}\parallel < \epsilon}$$
# 
# Покоординатная форма
# 
# $$x_1^{k+1} = (b_1-a_{12}x_2^k - ... - a_{1n}x_n^k)/a_{11}$$
# $$x_2^{k+1} = (b_2-a_{21}x_2^{k+1} - ... - a_{2n}x_n^k)/a_{22}$$
# $$...................................$$
# $$x_n^{k+1} = (b_n-a_{n1}x_1^{k+1} - ... - a_{nn-1}x_{n-1}^{k+1})/a_{nn}$$

# In[50]:


xk=np.zeros(n)
for i in range(n):
    xk[i]= 1
xk1 = np.zeros(n)
b = np.zeros(n)
for i in range(n):
    b[i]=1
eps = 0.01
norma = 1
m =0
Norm = list()
xr = list()
xx = list()
yy=[]
for i in range(0, n):
    xx.append(i)
while(norma>eps):
#     Заполняем координаты k+1 вектора
    H = 0
    for i in range(n):
        sum = 0
        for j in range(n):
            if(j <i):
                sum += M[i, j]*xk1[j]  
            if(j>i):
                sum += M[i,j]*xk[j]
        xk1[i] =(N[i] -sum)/M[i, i]
        H+= ((xk1[i]-xk[i]))**2
    norma = mt.sqrt(H)
    Norm.append(norma)
    xr.append(m+1)

    xk = copy.copy(xk1)
    xk1 = np.zeros(n)
    m+=1
    
    
    print(m)
# print(xk)
for i in range(n):
#     print(xk[i])
    yy.append(np.log10(np.abs(xk[i])))
#     print(yy[i])
plt.plot(xr, Norm)
plt.grid()
plt.show()
plt.plot(xx, yy)
plt.grid()
plt.show()


# Построим график решения X(i), а также график зависимости нормы решения от номера итерации предлагаемого метода численного решения системы.
# 
# Так же решим систему с помощью функции numpy.linalg.solve(…). Результат графически сравним с решением, полученным в пункте 3. 

# In[8]:


b = np.zeros(n)
for i in range(n):
    b[i]=1
prov = np.linalg.solve(M, N)
print(prov)


# In[39]:


plt.plot(xx, xk)
plt.plot(xx, prov)
plt.grid()
plt.show()

