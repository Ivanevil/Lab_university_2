l1 = [1]
s = 1

while sum(l1[0:s]) - sum(l1[0:s - 1]) > 0.001:
    s += 1
    l1 = [216 / 7 * (i**2 + 8 * i + 15) for i in range(s)]
l2 = [sum(l[0:k]) for k in range(s)]

print(s)
print(sum(l1))
print(l2)
