import pandas as pd

data = pd.read_csv('test.csv')
print(data)

data.to_pickle('test.pickle')