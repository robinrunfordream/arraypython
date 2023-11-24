import numpy as np

a =np.array([1,1,1])[:,np.newaxis]
b =np.array([2,2,2])[:,np.newaxis]

# c=np.vstack((a,b)) #上下合併 vertical stack
# d=np.hstack((a,b)) #左右合併 horizontal stack

# print(d)
# print(a.shape,c.shape,d.shape)

# print(a[:,np.newaxis].shape)

c = np.concatenate((a,b,b,a), axis=0)

print(c)