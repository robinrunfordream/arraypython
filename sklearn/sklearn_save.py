
from __future__ import print_function
from sklearn import svm
from sklearn import datasets
import os  # 新增這一行

# 檢查目錄是否存在，如果不存在則創建
# save_dir = 'save'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)



clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# method 1: pickle
import pickle
# save
# with open('save/clf.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# restore
with open('save/clf.pickle', 'rb') as f:
   clf2 = pickle.load(f)
   print(clf2.predict(X[0:1]))

# method 2: joblib
# from sklearn.externals import joblib #沒在使用了
import joblib  # 修改這一行
# Save
joblib.dump(clf, 'save/clf.pkl')
# restore
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))