import numpy as np
from sklearn import datasets
# from sklearn.cross_decomposition import tr

#from sklearn.cross_validation import train_test_split # 舊的引入方式
from sklearn.model_selection import cross_val_score # 新的引入方式
from sklearn.model_selection import train_test_split # 新的引入方式

from sklearn.neighbors import KNeighborsClassifier #鄰近預測值

iris = datasets.load_iris()
iris_x = iris.data #屬性存在DATA
iris_y = iris.target #分類


# print(iris_x[:2,:])
# print(iris_y) #3個類別的花

x_train,x_test,y_train,y_test=train_test_split(iris_x,iris_y,test_size=0.3)  #測試樣本 X_TEST Y_TEST 佔30%

# print(y_train)

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)
print(knn.predict(x_test)) #用MODEL去預測值
print(y_test) #真實數據