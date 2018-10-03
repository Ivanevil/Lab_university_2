def product(val, n):
    mul = 1
    for i in range(n):
        if i:
            mul *= val - x[i - 1]
        yield mul


x = [25, 52, 33, 42, 24]
y = [22, 12, 12, 25, 22]
c = []
for n in range(len(x)):
    p = product(x[n], n + 1)
    c.append((y[n] - sum(c[k] * next(p) for k in range(n))) / next(p))
    print(c)


def f(v):
    return sum(c[k] * p for k, p in enumerate(product(v, len(c))))


n = 1
for i in range(3):
    if i:
        n *= x[2] - x[1]
    print(i, n)
