from __future__ import print_function
from sklearn import preprocessing #標準化
import numpy as np
from sklearn.model_selection import train_test_split #分開模塊
# from sklearn.datasets.samples_generator import make_classification#沒有在用了
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification  # 更新導入語句

#===標準化，讓機械學習更容易處理資料 
# a = np.array([[10, 2.7, 3.6],
#                      [-100, 5, -2],
#                      [120, 20, 40]], dtype=np.float64)
# print(a)
# print(preprocessing.scale(a))

X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X = preprocessing.scale(X)    # normalization step

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))