# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:40:18 2018

@author: leedom
"""


from scipy.stats.stats import pearsonr   
a = [21332,20162,19138,18621,18016] #算法工程师
b = [13854,12213,11009,10655,9503] #所有程序员
print(pearsonr(a,b))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
X = [[13854], [12213], [11009], [10655], [9503]]
y =  [21332, 20162, 19138, 18621, 18016]
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)
print(lr.score(X,y)) #r^2

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 
#绘制图形时使用了中文标题，出现了乱码，原因是matplotlib.pyplot在显示时无法找到合适的字体
#先把需要的字体（在系统盘c盘的windows下的fonts目录内添加到FontProperties中）
cities = [u'北京',u'上海',u'杭州',u'深圳',u'广州']
plt.scatter(X, y,  color='black')
plt.plot(X, lr.predict(X), color='y',
         linewidth=3)
plt.title(u"程序员和算法工程师平均工资",fontproperties=font_set)
plt.xlabel(u"程序员平均工资",fontproperties=font_set)
plt.ylabel(u"算法工程师平均工资",fontproperties=font_set)
for i in range(5):
    plt.text(x=X[i][0],y=y[i], s=cities[i],fontproperties=font_set, color='b')
plt.show()