import random
import sys
sys.setrecursionlimit(100000)


def w(C):
    B = 4384800480252860265134886651572686223217614265308437613400946197267225928506988971438690 * random.randint(100, 1000) * random.randint(100, 1000) * random.randint(100, 1000) * random.randint(100, 1000) * random.randint(1, 10) - 1
    if len(str(B)) == C:
        return B
    else:
        return w(C)


def toBinary(n):
    r = []
    while (n > 0):
        r.append(n % 2)
        n = n / 2
        return r


def MillerRabin(n, s=50):
    for j in range(1, s + 1):
        a = random.randint(1, n - 1)
        b = toBinary(n - 1)
        d = 1
        for i in range(len(b) - 1, -1, -1):
            x = d
            d = (d * d) % n
            if d == 1 and x != 1 and x != n - 1:
                return True  # Составное
            if b[i] == 1:
                d = (d * a) % n
                if d != 1:
                    return True  # Составное
                return False  # Простое


def func(X):
    A = w(X)
    if MillerRabin(A) == 0:
        return A
    else:
        return func(X)


X = int(input('Your Value '))
print(func(X))
