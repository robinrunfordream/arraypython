from sklearn import datasets 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
data_x = housing.data
data_y = housing.target

#=================================== 可以跑
model = LinearRegression()

model.fit(data_x, data_y) #默認的參數

# print(model.predict(data_x[:4, :]))  #訓練完的MODEL來預測值
# print(data_y[:4]) #真實數據


# print(model.coef_) #y=0.1X + 0.3 ,0.1   x參數
# print(model.intercept_)  #0.3， Y軸交點
# print(model.get_params()) #定義那些參數
print(model.score(data_x, data_y)) # R^2 coefficient of determination  X預測 Y對比

#===================================


# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
# plt.scatter(X, y)
# plt.show()