# _*_ coding utf-8 _*_
# @Author : 段钰
# @Email : duanyu@bjtu.edu.cn
# @Time : 2022/3/21 16:15 
# @IDE: PyCharm

"""
1.	编程实现k近邻算法对sklearn库中的乳腺癌数据集进行分类，需列表比较不同k值对结果的影响。
"""
import sys
print('Python %s on %s' % (sys.version, sys.platform))


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import operator

import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data['data']
y = data['target']


# 数据载入到此为止


class KNN_Model(object):
    def __init__(self, trainData, testData, labels, k):
        self.trainData = trainData
        self.testData = testData
        self.labels = labels
        self.k = k

    def run(self):
        rowSize = self.trainData.shape[0]
        # 计算训练样本和测试样本的差值
        diff = np.tile(self.testData, (rowSize, 1)) - self.trainData
        # 计算差值的平方和
        sqrDiff = diff ** 2
        sqrDiffSum = sqrDiff.sum(axis=1)
        # 计算距离
        distances = sqrDiffSum ** 0.5
        # 对所得的距离从低到高进行排序
        sortDistance = distances.argsort()
        # 从小到大排序，并返回索引值
        count = {}

        for i in range(self.k):
            vote = self.labels[sortDistance[i]]
            count[vote] = count.get(vote, 0) + 1
        # 对类别出现的频数从高到低进行排序
        sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

        # 返回出现频数最高的类别
        return sortCount[0][0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

result = []
res_stat = {}
for k in range(1, 101):
    result.append([])
    for sample_index in range(len(X_test)):
        session = KNN_Model(X_train, X_test[sample_index], y_train, k)
        result[k - 1].append(session.run())
    result[k - 1] = (np.array(result[k - 1]) == np.array(y_test))
    res_stat['k=' + str(k)] = sum(result[k - 1] / len(result[k - 1]))

print(res_stat)

x_points = np.array([i for i in range(1,101)])
y_points = np.array([i[1] for i in res_stat.items()])

plt.plot(x_points, y_points,marker = 'o')
plt.xlabel("the Value of K")
plt.ylabel("Accuracy")


plt.show()
