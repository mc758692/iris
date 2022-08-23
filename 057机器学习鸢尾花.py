#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:55
# @Author  : 奕凌天
# @Site    : 
# @File    : 057机器学习鸢尾花.py
# @Software: PyCharm
# @href    : https://www.cnblogs.com/linlf03/p/16117231.html

# 安装库
"""
pip install matplotlib
pip install scipy
pip install numpy
pip install pandas
pip install scikit-learn
"""

# 模块引入
# import sys
# import scipy
# import numpy
# import matplotlib
# import pandas
# import sklearn

# 检查模块版本
# print('Python: {}'.format(sys.version))
# print('scipy: {}'.format(scipy.__version__))
# print('numpy: {}'.format(numpy.__version__))
# print('matplotlib: {}'.format(matplotlib.__version__))
# print('pandas: {}'.format(pandas.__version__))
# print('sklearn: {}'.format(sklearn.__version__))


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris


# 下载鸢尾花数据
data = load_iris()
outputfile = "iris.csv"  # 保存文件路径名
column = list(data['feature_names'])
dd = pandas.DataFrame(data.data, index=range(150), columns=column)
dt = pandas.DataFrame(data.target, index=range(150), columns=['outcome'])

jj = dd.join(dt, how='outer')  # 用到DataFrame的合并方法，将data.data数据与data.target数据合并

jj.to_csv(outputfile, header=False, index=False)  # 将数据保存到outputfile文件中
# jj.to_csv(outputfile)

print("------------------------------------------------")



# Load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)
dataset = read_csv("F:\\孟成资料\\Python学习\\实践\\057机器学习鸢尾花\\data\\iris.csv", names=names)
# print(dataset)

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

# classdistribution
print("------------------------------------------------")
print(dataset.groupby('class').size())

# boxand whisker plots
print("------------------------------------------------")
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

print("------------------------------------------------")
# histograms
# dataset.hist()
# pyplot.show()


# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()


# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
