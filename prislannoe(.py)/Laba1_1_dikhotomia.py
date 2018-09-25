
# coding: utf-8

# In[ ]:


a = 0
b = 0
eps = 0.001


def f(x):
    return 3 * x**4 + 4 * x**3 - 12 * x**2 - 7


for i in range(-50, 50):
    if (f(i) * f(i + 1) < 0):
        a = i
        b = i + 1
print(a, b)
c = (a + b) / 2
y = f(c)
while(abs(y) > eps):
    c = (a + b) / 2
    y = f(c)
    if(f(a) * y < 0):
        b = c
    else:
        a = c
print(c)

