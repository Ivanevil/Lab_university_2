l = []
n = 1000
s = 3
for i in range(2, n + 1):
    k = 0
    for j in range(2, i):
        if i % j == 0:
        k += 1
    if k == 0:
        l.append(i)
l_new = l[::s]
print(l_new)
