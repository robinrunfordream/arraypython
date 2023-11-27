import pandas as pd
import numpy as np


dates = pd.date_range('20130101', periods=6)
df =pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d']) #指定格式


df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan


print(df)
print(np.any(df.isnull()==True)) #資料表太大有缺失值，快速查詢辦法
print(df.isnull())#確認資料是否有缺失的值
# print(df.fillna(value=0)) #nan 替換成0

# print(df.dropna(axis=0, how='any')) #一行值有NAN就丟，how ={'any' ,all} 整行都NAN才丟

