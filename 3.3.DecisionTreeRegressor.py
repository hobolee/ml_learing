#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    N = 100
    x = np.random.rand(N) * 6 - 3     # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print(y)
    x = x.reshape(-1, 1)
    print(x)

    # model = DecisionTreeRegressor(criterion='mse', max_depth=10)
    model = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=10)
    model.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = model.predict(x_test)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(x, y, 'r*', markersize=10, markeredgecolor='k', label='实际值')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='预测值')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树/随机森林回归', fontsize=15)
    plt.tight_layout(2)
    plt.show()

    # 比较决策树的深度影响
    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    # model = DecisionTreeRegressor(criterion='mse')
    model = RandomForestRegressor(n_estimators=20, criterion='mse')
    plt.figure(facecolor='w')
    plt.plot(x, y, 'ro', ms=5, mec='k', label='实际值')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    for d, c in zip(depth, clr):
        model.set_params(max_depth=d)
        model.fit(x, y)
        y_hat = model.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=c, linewidth=2, markeredgecolor='k', label='Depth=%d' % d)
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树/随机森林回归', fontsize=15)
    plt.tight_layout(2)
    plt.show()
