import numpy as np
import math

def f(x,y): return sin(x-y) - x*y + 1
def f1x(x,y): return cos(x-y) - y
def f1y(x,y): return -cos(x-y) - x

def g(x,y): return x**2 + y**2 -1
def g1x(x,y): return 2*x
def g1y(x,y): return 2*y

def newt(x0, y0, f, g, eps): 
    x0 = float(x0)
    y0 = float(y0)
    A = np.zeros((2,2))
    while True:
        A[0, 0] = f1x(x0, y0)
        A[0, 1] = f1y(x0, y0)
        A[1, 0] = g1x(x0, y0)
        A[1, 1] = g1y(x0, y0)
      
        B=np.linalg.inv(A)
        
        x1 = x0 - f(x0, y0)*B[0, 0] -g(x0, y0)*B[0, 1]
        y1 = y0 - f(x0, y0)*B[1, 0] - g(x0, y0)*B[1, 1]
        
        if (abs(x1 - x0) and abs(y1 - y0)) < eps: 
            return x1, y1
        x0 = x1
        y0 = y1




print(newt(5,f,f1,0.0001))