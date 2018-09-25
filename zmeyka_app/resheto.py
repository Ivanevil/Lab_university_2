# Листинг 8
n = int(input("n="))
a = range(n + 1)
lst = []

i = 2
while i <= n:
    if a[i] != 0:
        lst.append(a[i])
        for j in range(i, n + 1, i):
            i += 1
print(lst)
