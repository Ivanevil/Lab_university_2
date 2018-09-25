
# coding: utf-8

# In[1]:


a = 0
b = 0
eps = 0.0001


def f(x):
    return 3 * x**4 + 4 * x**3 - 12 * x**2 - 7


def fx(x):
    return 12 * x**3 + 12 * x**2 - 24 * x


def fxx(x):
    return 36 * x**2 + 24 * x - 24


def xn(a, b):
    return a - f(a) * (b - a) / (f(b) - f(a))


def xnn(a, b):
    return b - f(b) * (b - a) / (f(b) - f(a))


for i in range(-50, 50):
    if (f(i) * f(i + 1) < 0):
        a = i
        b = i + 1
print(a, b)
c = (a + b) / 2
print(fx(c), fxx(c))
if(fx(c) * fxx(c) > 0):
    x1 = xn(a, b)
    a = x1
    x2 = xn(a, b)
    while (abs(x2 - x1) > eps):
        x1 = x2
        a = x2
        x2 = xn(a, b)
    print(x1)
else:
    x1 = xn(a, b)
    a = x1
    x2 = xn(a, b)
    while (abs(x2 - x1) > eps):
        x1 = x2
        a = x2
        x2 = xn(a, b)
    print(x1)

