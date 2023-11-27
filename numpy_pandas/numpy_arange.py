import numpy as np 
# a = np.arange(2,14).reshape((3,4))


# print(a)
# print(np.argmin(a))
# print(np.argmax(a))
# print(np.median(a))
# print(np.cumsum(a)) #累加
# print(np.diff(a))   #相減

b = np.arange(14,2,-1).reshape((3,4))
print(b)
print(np.sort(b))
print(np.transpose(b))#矩陣反向
print(b.T)#矩陣反向
print((b.T).dot(b))
print(np.clip(b,5,9)) #小於5 =5 大於9 =9