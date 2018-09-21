import math 

eps= 0.0001
"""
def f(x): 
    return math.sin(x-math.sqrt(x**2-1))-x*math.sqrt(x**2-1)+1

# l1 = [i for i in range(-10, 10, 1) if f(i) * f(i - 1) < 0] 
# print(l1) для метода подстановки


def f1(x):
    return (1-2*x**2+(math.sqrt(x**2-1)-x)*math.cos(x - math.sqrt(x**2-1)))/(math.sqrt(x**2-1))
"""

def f(x,y):
    return math.sin(x-y)-x*y+1
    
def f1(x,y):
    return math.cos(x-y)-y-math.cos(x-y)-x


def g(x, y):
    return x**2-y**2-1

def g1(x, y):
    return 2*x-2*y


def h(x, y):
    return x**2 + y**2 - 6

def h1(x, y):
    return 2*x+2*y


"""
#для способа с подстановкой
def method1(a, b):
    x0 = (a + b) / 2
    xn = f(x0) 
    xn1 = xn - f(xn) / f1(xn) 
    n = 0 
    while abs(xn1 - xn) > eps: 
        xn = xn1 
        xn1 = xn - f(xn) / f1(xn)
        n += 1 
        
    y0 = (a + b) / 2
    yn = g(xn1, y0) 
    yn1 = yn - g(xn1, yn) / g1(xn1,yn) 
    while abs(yn1 - yn) > eps: 
        yn = yn1
        yn1 = yn - g(xn1,yn) / g1(xn1,yn)
        n += 1
    return xn1, -yn1, n
"""
    
# без подстановки
def method2(a, b,hunc,hunc1,gunc,gunc1):
    x0 = (a + b) / 2
    y = 0
    
    xng = gunc(x0, y) 
    xng1 = xng - gunc(xng, y) / gunc1(xng,y)
    
    xnh = hunc(x0, y)
    xnh1 = xnh - hunc(xnh, y) / hunc1(xnh, y)
    

    while abs(xnh1)>abs(xng1):
        xng = gunc(x0, y) 
        xng1 = xng - gunc(xng, y) / gunc1(xng,y)
    
        while abs(xng1 - xng) > eps: 
            xng = xng1
            xng1 = xng - gunc(xng,y) / gunc1(xng,y)
            y += eps
           # нашли пересчение при у=0 для g 
 
        xnh = hunc(x0, y)
        xnh1 = xnh - hunc(xnh, y) / hunc1(xnh, y)
        while abs(xnh1 - xnh) > eps: 
            xnh = xnh1
            xnh1 = xnh - hunc(xnh,y) / hunc1(xnh,y)
            y += eps
           # нашли пересечение при у=0 для h
        
    return -xng1, y


"""
#работает для любой заданной функции как метод1
def method3(a, b, func, func1,y):
    x0 = (a + b) / 2
    xn = func(x0,y) 
    xn1 = xn - func(xn,y) / func1(xn,y) 
    
    while abs(xn1 - xn) > eps: 
        xn = xn1 
        xn1 = xn - func(xn,y) / func1(xn,y)
     
       
    return xn1

"""    
def final(a,b,func,func1,gunc,gunc1):
    (s, k) = method2(a,b,func,func1,gunc,gunc1)
    if k <= 0.1:
        (s, k) = method2(a,b,gunc,gunc1,func,func1)
    if k <= 0.1:
       return print("net peresechenii")            
    return (s, k)
              
        
    
    


print(final(0,5,f,f1,g,g1))