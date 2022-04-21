# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:17 
# @IDE: PyCharm

'''
4.	编程实现SVM算法对乳腺癌数据集进行分类，并自选一种核函数与无核函数的情况进行比较。
'''

import sys

print('Python %s on %s' % (sys.version, sys.platform))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

session = svm.SVC(kernel='linear')
session.fit(X_train, y_train)

y_pred = session.predict(X_test)
report = sm.classification_report(y_test, y_pred)
print("\nTraining Report:\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("While using Linear Kernel:")
print(report)

acc = session.score(X_test, y_test)
print('Accuracy:', acc)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print("While using rbf Kernel:")
session = svm.SVC(kernel='rbf')
session.fit(X_train, y_train)

y_pred = session.predict(X_test)
report = sm.classification_report(y_test, y_pred)

print(report)
acc = session.score(X_test,y_test)
print('Accuracy:', acc)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


