from sklearn import datasets 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
data_x = housing.data
data_y = housing.target

# loaded_data = datasets.load_boston() #舊版 被提除了

# data_x = loaded_data.data
# data_y = loaded_data.target

#=================================== 可以跑
# model = LinearRegression()

# model.fit(data_x, data_y) #默認的參數

# print(model.predict(data_x[:4, :]))  #訓練完的MODEL來預測值
# print(data_y[:4]) #真實數據
#===================================


X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, y)
plt.show()