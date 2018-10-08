import numpy as np
import matplotlib.pyplot as plt

fid = open("./data_var16.txt", "r")
lines = fid.readlines()
fid.close


def fin_dif(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])
    return a


def inter(x1, y1, r):
    a = fin_dif(x1, y1)
    n = len(x1) - 1  # степень полинома
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (r - x1[n - k]) * p
    return p


x = np.array([1, 2, 3, 4, 5])
y = []
for i in range(1, 6):
    y.append(float(lines[i].split(',')[1]))

x_arr = np.linspace(np.min(x), np.max(x), 40)
y_arr = [inter(x, y, i) for i in x_arr]

fig = plt.figure(figsize=(25, 13))
ax = fig.gca()
ax.set_xticks(np.arange(0, 11, 0.1))
ax.set_yticks(np.arange(0, 30, 0.1))
plt.scatter(x, y)
plt.grid()
plt.plot(x, y, 'o', x_arr, y_arr)
plt.show()
