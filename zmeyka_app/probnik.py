def f(x):
    return x**2 - 4 * x - 18 - x**3


eps = 0.0001


def formula(a, b):
    return f(a) - f(b) - 0.2 * (f(b)**2 + 2 * f(b) - 1)


def itr(a, b, imax):
    for i in range(imax):
        a = formula(i, i + 1)
        b = formula(i, i + 1)
        print(b)
        if abs(f(a) - f(b)) < eps:
            break

    return a, b


print(itr(0, 2, 2000))
