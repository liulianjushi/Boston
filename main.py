import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import sklearn.datasets as datasets

# 机器算法模型
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 切割训练数据和样本数据
from sklearn.model_selection import train_test_split

# 用于模型评分
from sklearn.metrics import r2_score

'''start 生成训练数据和测试数据'''
boston = datasets.load_boston()
train = boston.data
target = boston.target

plt.figure()
plt.plot(target)
plt.show()

# 切割数据样本集合测试集
X_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2)

'''end 生成训练数据和测试数据'''

knn = KNeighborsRegressor()
linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
decision = DecisionTreeRegressor()
svr = SVR()

knn.fit(X_train, y_train)
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
decision.fit(X_train, y_train)
svr.fit(X_train, y_train)

y_pre_knn = knn.predict(x_test)
y_pre_linear = linear.predict(x_test)
y_pre_ridge = ridge.predict(x_test)
y_pre_lasso = lasso.predict(x_test)
y_pre_decision = decision.predict(x_test)
y_pre_svr = svr.predict(x_test)

knn_score = r2_score(y_test, y_pre_knn)
linear_score = r2_score(y_test, y_pre_linear)
ridge_score = r2_score(y_test, y_pre_ridge)
lasso_score = r2_score(y_test, y_pre_lasso)
decision_score = r2_score(y_test, y_pre_decision)
svr_score = r2_score(y_test, y_pre_svr)
print(knn_score, linear_score, ridge_score, lasso_score, decision_score, svr_score)

# KNN
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_knn, label='knn')
plt.legend()

# Linear
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_linear, label='linear')
plt.legend()

# Ridge
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_ridge, label='ridge')
plt.legend()

# lasso
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_lasso, label='lasso')
plt.legend()

# decision
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_decision, label='decision')
plt.legend()

# SVR
plt.figure()
plt.plot(y_test, label='true')
plt.plot(y_pre_svr, label='svr')
plt.legend()
plt.show()
