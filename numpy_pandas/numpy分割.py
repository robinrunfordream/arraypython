import numpy as np
a = np.arange(12).reshape((3,4))
print(a)
# print(np.split(a,3,axis=0)) #等項分割

# print(np.array_split(a,3,axis=1)) #不等項分割

print(np.vsplit(a,3))
print(np.hsplit(a,2))