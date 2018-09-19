# metod delenia popolam (dihotomii)

def f(x):
    return 3*x**4+4*x**3-12*x**2-7

l = [i for i in range(-10, 10, 1) if f(i)*f(i-1)<0]
print(l)

eps = 0.00001

def MDP(f, a, b):
    n = 0
    while abs(b-a) >= eps:
        c = (b+a)/2
        if f(c)==0:
            break
        elif f(a)*f(c)<0:
            b=c
        else:
            a=c
        n += 1
    return c, f(c), n

print(MDP(f, -2, 2))

# metod hord iz wiki

def formula(a, b):
    return a - (f(a)*(b-a))/(f(b)-f(a))
    
def korni(a, b):
    n = 0
    while abs(b-a)>=eps:
        a = formula(a, b)
        b = formula(a, b)
        n += 1
    return b, f(b), n
    
print(korni(-2, 2))
    