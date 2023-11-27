import numpy as np

a = np.arange(3,15).reshape((3,4))

print(a)


# for row in a:
#     print(row)
# print(a.T)
# for column in a.T:
#     print(column)

print(a.flatten())
for item in a.flat:
    print(item)
