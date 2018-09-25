
# coding: utf-8

# In[ ]:


import scipy as sp
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import pylab
from matplotlib import mlab

def f(t,p):
    n=np.sin(t)*np.exp(- p**2 *t**2)
    return n

def simpson(a,b,p):
    fla=((b-a)/90)*(7*f(a, p) + 32*f((3*a+b)/4, p) + 12*f((a+b)/2, p) + 32*f((a+3*b)/4, p)+7*f(b, p))
    return (fla)

def rims(i):
    mas=[]
    it1=simpson(0,1,i)
    mas.append(it1)
    it2=simpson(0,0.5,i)+simpson(0.5,1,i)
    mas.append(it2)
    j=2
    while (abs(it1/it2-1)>0.01):
        j+=1
        it1=it2
        it2=0
        for k in range (j):
            it2+=simpson(k/j,(k+1)/j,i)
        mas.append(it2)
    #print (j)
    return (mas)

    #return (it2)

mas=rims(100)
print(mas)
print(len(mas))
dlina=len(mas)

int=[]
int.append(0)
i=1
while i<100:
    mas=rims(i)
    int.append(mas[len(mas)-1])
    i+=1


tlist = mlab.frange (0, 1, 0.001)
ylist = [f (t,2) for t in tlist]
plt.xlabel('x')
plt.ylabel('y')
plt.title('График подынтегральной функции')
pylab.plot (tlist, ylist)
plt.grid(True)
pylab.show()


xlist = mlab.frange (1,dlina, 1)
ylist = mas
#plt.axis([-10, 10, -10,10])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Значение интеграла от номера итерации')
pylab.plot (xlist, ylist)
plt.grid(True)
pylab.show()



## Лабораторная работа №5
Численное решение системы линейных уравнений, численное интегрирование

** Цель работы: **
научиться считать определенные интеграллы и решать системы линейных уравнений численными методами и с помощью встроенных библиотек.

** Вычисление интеграла :  **
используем квадратурную формулу Буля(n=4)
$$ \sum^{4}_{k=0}C^{(4)}_{k}f(x_{k}^{(4)}) = \frac{(\beta - \alpha)}{90}(7f(\alpha) + 32f(\frac{3\alpha +\beta}{4}) + 12f(\frac{\alpha +\beta}{2}+ 32f(\frac{\alpha +3\beta}{4}) + 7f(\beta)) $$


import numpy as np
import math as mt
n=100
M=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        M[i,j] = -(j+4)*mt.sin(j)/((1+i**2)*(2+j**4))
    M[i,i] += n**2
np.set_printoptions(suppress=True, precision =4, linewidth=100)
print(M)
import numpy as np
import math as mt
import copy
import matplotlib.pyplot as plt
tau = 1
xk=np.zeros(n)
for i in range(n):
    xk[i]= 1
xk1 = np.zeros(n)
b = np.zeros(n)
for i in range(n):
    b[i]=i
xx = list()
for i in range(n):
    xx.append(i)
eps=0.001
norma=1
Norm = list()
normax = list()
mx = list()
m = 0
while(norma>eps):
    H=0
    for i in range(n):
        sum = int[i]
        for j in range(n):
            sum -=M[i,j]*xk[j]
        xk1[i]=xk[i]+(tau/M[i,i])*sum
        H+= ((xk1[i]-xk[i]))**2
    norma = mt.sqrt(H)
    Norm.append(norma)
    xk = copy.copy(xk1)
    xk1 = np.zeros(n)
    m+=1
    mx.append(m)
    sumx =0
    for i in range(n):
        sumx+=xk[i]**2
    normax.append(mt.sqrt(sumx))
plt.plot(xx, xk)
plt.grid()
plt.show()
plt.plot(mx, normax)
plt.grid()
plt.show()
import numpy as np
# b = np.zeros(n)
# for i in range(n):
#     b[i]=1
prov = np.linalg.solve(M, int)
# print(prov)
plt.plot(xx, np.log10(np.abs(xk-prov)))
# plt.plot(xx, prov)
plt.grid()
plt.show()



$${\large Задание 4 и 4+}$$

Решить для n=1000 и n=10000 методом наискорейшего спуска систему:
$$n^{2}x_i = \sum_{j=1}^{n}\frac{j+4sin(j)}{1+i^2+j^4}*j*x_j + int_0^{\infty}\sin(t)exp(-i*t^2)dt, i=\overline{1,n}$$

Метод последовательной релаксации

Итерационными называются приближенные методы, в которых решение системы получается как предел последовательности векторов $$(x^k)_{k=1}^{\infty}$$, каждый последующий элемент которой вычисляется по некоторому единому правилу. Начальный элемент $$x^1$$ выбирается произвольно.
Условие сходимости
$${\parallel\frac{x^{k+1}-x^{k}}{x^k}\parallel < \epsilon}$$

Как правило, для итерационного метода решения системы существует такая последовательность невырожденных матиц $$H_k$$, что правило мостроения элементов итерационной последовательности записывается в виде $$x^{k+1} = x^k-H_k(Ax^k-b)$$ или $$x^{k+1} = T_kx^k+H_kb,$$ где $$T_k=E-H_kE,$$ Е - единичная матрица nxn

В методе последовательной релаксации $$H_k={\tau}(D+{\tau}L)^-1$$ $$T_k=(D+{\tau}L)^-1((1-{\tau})D-{\tau}R)$$
$$0<{\tau}<2$$

Тогда итерационный процесс имеет вид$$x^{k+1} = (D+{\tau}L)^-1((1-{\tau})D-{\tau}R)x^k+{\tau}(D+{\tau}L)^-1)b$$


$${\large Задание 5 и 5+}$$

Построить график решения X(i), а также график зависимости нормы решения от номера итерации предлагаемого метода численного решения системы.

Решить систему с помощью функции numpy.linalg.solve(…). Результат графически сравнить с решением, полученным в пункте 3. Разницу объяснит.

