# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:17 
# @IDE: PyCharm

'''
5.	编程实现朴素贝叶斯算法对sklearn库中的乳腺癌数据集进行分类。
'''

import sys
import numpy as np
print('Python %s on %s' % (sys.version, sys.platform))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as sm
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

session = GaussianNB().fit(X_train, y_train)
y_pred = session.predict(X_test)
accuracy = session.score(X_test, y_test)
report = sm.classification_report(y_pred, y_test)

print("\nTraining Report:\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(report)

acc = session.score(X_test, y_test)
print('Accuracy:', acc)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
