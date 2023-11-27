import pandas as pd
import numpy as np


#concatenation

# df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
# df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
# df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])


# # print(df1)
# # print(df2)
# # print(df3)
#==================================
# res = pd.concat([df1, df2, df3], axis=0, ignore_index= True)

# print(res)
# join,['inner', 'outer']
# df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
# df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
# print(df1)
# print(df2)
# res = pd.concat([df1, df2], join='inner',ignore_index=True) #inner只保留兩者都有的部分
# print(res)
#==================================

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
res = pd.concat([df1, df2], axis=1).reindex(df1.index)
print(res)
#==================================

