def is_prime(number):
    """Эту функцию можно сильно оптимизировать. Подумайте, как"""

    if number == 1:
        return False  # 1 - не простое число

    for i in range(2, number):
        if number % i == 0:
            return False

    return number


print(is_prime(10000))

# l = []
# n = 10
# s = 1
# for i in range(2, n + 1):
#     k = 0
#     for j in range(2, i):
#         if i % j == 0:
#             k += 1
#     if k == 0:
#         l.append(i)
# l_new = l[::s]

l_new2 = [is_prime(i) for i in range(10000) if is_prime(i) != False]
l_new3 = l_new2[::3]
# print(l_new)
print(l_new3)
