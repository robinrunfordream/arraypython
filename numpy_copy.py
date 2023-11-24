import numpy as np 
a = np.arange(4)
print(a)
b = a
c = a
d = b
a[0] = 11

print(a)
print(b is a)
print(d is a)
d[1:3]= [22,33]
print(a,b,d)

b = a.copy() #deep copy ，只想要值 不要關聯

a[0] = 44
print(b)