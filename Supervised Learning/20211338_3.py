# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:16 
# @IDE: PyCharm

'''
3.	编程实现线性模型对乳腺癌数据集进行分类。
'''

import sys

print('Python %s on %s' % (sys.version, sys.platform))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

session = LinearRegression()
session.fit(X_train, y_train)
print('\n==========================================================================')
print("W1到Wn为：", session.coef_)
print("W0为：", session.intercept_)
print("模型正确率：", session.score(X_test, y_test))
print('==========================================================================')