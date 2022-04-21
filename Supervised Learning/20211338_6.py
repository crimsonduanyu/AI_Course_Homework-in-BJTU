# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:17 
# @IDE: PyCharm

'''
6.	选择sklearn库中的鸢尾花数据集或者自选与专业相关的数据集，
采用三种以上的监督学习算法进行分类，选择你认为合适的模型评估参
数（错误率、查准率、查全率、F1等）对这些算法进行评估对比，选出
最优模型。
'''
import sys
import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
print()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#   决策树算法
from sklearn import tree
import os
import pydotplus

session = tree.DecisionTreeClassifier(criterion='gini')
session = session.fit(X_train, y_train)

accuracy = session.score(X_test, y_test)

feature_name = data['feature_names']

dot_data = tree.export_graphviz(session, out_file=None,
                                feature_names=feature_name,
                                class_names=data['target_names'])

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(r'.\6题决策树PDF可视化\Original.pdf')

path = session.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print('ccp_alphas:', ccp_alphas)
print('impurities', impurities)

print("未剪枝时，正确率为:", accuracy)
acc_stat = []
acc_stat.append(accuracy)

for i in range(1, len(ccp_alphas)):
    session = tree.DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alphas[i] + 1e-6)
    session = session.fit(X_train, y_train)
    accuracy = session.score(X_test, y_test)
    acc_stat.append(accuracy)
    print("===================================================")
    print("设置alpha={}\n剪枝后的树不纯度：{}\n此时的模型准确率：{}".
          format(ccp_alphas[i], impurities[i], accuracy))
    dot_data = tree.export_graphviz(session, out_file=None,
                                    feature_names=feature_name,
                                    class_names=data['target_names'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(r'.\6题决策树PDF可视化\CUT' + str(i) + '.pdf')

print('各步骤的剪枝决策树可视化图已经保存到： \n\\AI_Course_HW\\6题决策树PDF可视化\\')

the_index = np.array(acc_stat).argsort()[-1]
print(
    "其中alpha={}时，树不纯度为{}，有最高准确率{}，此时对应可视化图{}".format(ccp_alphas[the_index], impurities[the_index], acc_stat[the_index],
                                                     the_index + 1))
model_result_dict = {}
model_result_dict['tree'] = acc_stat[the_index]

plt.plot(ccp_alphas, acc_stat, marker='o')
plt.xlabel("Value of CCP_ALPHA")
plt.ylabel("Accuracy")
plt.show()

#   决策树算法到此为止


#   线性模型分类算法
from sklearn.linear_model import LinearRegression

session = LinearRegression()
session.fit(X_train, y_train)
print('\n==========================================================================')
print("W1到Wn为：", session.coef_)
print("W0为：", session.intercept_)
accuracy = session.score(X_test, y_test)
print("模型正确率：", accuracy)
print('==========================================================================')
model_result_dict['linear_model'] = accuracy

#   线性模型分类算法到此为止

#   SVM算法
import sklearn.svm as svm
import sklearn.metrics as sm

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
model_result_dict['svm_linear'] = acc
print("While using rbf Kernel:")
session = svm.SVC(kernel='rbf')
session.fit(X_train, y_train)

y_pred = session.predict(X_test)
report = sm.classification_report(y_test, y_pred)

print(report)
accuracy = session.score(X_test, y_test)
print('Accuracy:', accuracy)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
model_result_dict['svm_rbf'] = accuracy

#   SVM算法到此为止

#   朴素贝叶斯算法
from sklearn.naive_bayes import GaussianNB

session = GaussianNB().fit(X_train, y_train)
y_pred = session.predict(X_test)
report = sm.classification_report(y_pred, y_test)

print("\nTraining Report:\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(report)
accuracy = session.score(X_test, y_test)
print('Accuracy:', accuracy)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
model_result_dict['naive_Bayers'] = accuracy

# 朴素贝叶斯算法到此为止

print(model_result_dict)
for i in sorted(model_result_dict.items(), key=lambda item: item[1], reverse=True):
    print('Model', i[0], 'has accuracy', i[1])

print("The best Model is:", sorted(model_result_dict.items(), key=lambda item: item[1], reverse=True)[0][0])
