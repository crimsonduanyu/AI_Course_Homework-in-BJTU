# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:16 
# @IDE: PyCharm

'''
2.	编程实现决策树算法对乳腺癌数据集进行分类，试比较剪枝与不剪枝对结果的影响。
'''
import sys

print('Python %s on %s' % (sys.version, sys.platform))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import os
import pydotplus

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

session = tree.DecisionTreeClassifier(criterion='gini')
session = session.fit(X_train, y_train)

accuracy = session.score(X_test, y_test)

feature_name = data['feature_names']

dot_data = tree.export_graphviz(session, out_file=None,
                                feature_names=feature_name,
                                class_names=['malignant', 'benign'])

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(r'.\2题决策树PDF可视化\Original.pdf')

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
                                    class_names=['malignant', 'benign'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(r'.\2题决策树PDF可视化\CUT' + str(i) + '.pdf')

print('各步骤的剪枝决策树可视化图已经保存到： \n\\AI_Course_HW\\2题决策树PDF可视化\\')
print("Prosess Terminated")

plt.plot(ccp_alphas, acc_stat, marker='o')
plt.xlabel("Value of CCP_ALPHA")
plt.ylabel("Accuracy")
plt.show()
