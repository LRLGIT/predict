import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
data =  pd.read_csv('ex1data2.txt',names = ['Size','Bedrooms','Price'])#文件路径
data.head()#查看前五行

#可视化数据
data.plot.scatter('Size', 'Price', label='Size')  # 画出房间大小与价格数据集散点图
plt.show()
data.plot.scatter('Bedrooms', 'Price', label='Size')  # 画出卧室数量大小与价格数据集散点图
plt.show()

#均值归一化
data=(data-data.mean())/data.std()
data.head()

#数据处理
data.insert(0,'ones',1)# 在数据集中插入第一列，列名为ones,数值为1
data.head()
col = data.shape[1]
X = np.array(data.iloc[:, 0:col - 1])
Y = np.array(data.iloc[:, col - 1:col])
#初始化参数theta
theta = np.zeros((3, 1))  # 将theta初始化为一个（3，1）的数组


# 损失函数
def Lost(x, y, theta):
    i = np.power(x @ theta - y, 2)
    return np.sum(i) / (2 * len(x))



#梯度下降算法
def gradientDescent(x,y,theta,counts):
    costs = []#创建存放总损失值的空列表
    for i in range(counts):#遍历迭代次数
        theta = theta - x.T@(x@theta-y)*alpha/len(x)
        cost = Lost(x,y,theta)#调用损失函数得到迭代一次的cost
        costs.append(cost)#将cost传入costs列表
        if i%100 == 0:    #迭代100次，打印cost值
            print(cost)
    return theta,costs


alpha_list = [0.003,0.03,0.0001,0.001,0.01]  #设置alpha
counts = 200  #循环次数


fig, ax = plt.subplots()
for alpha in alpha_list:  # 迭代不同学习率alpha
    _, costs = gradientDescent(X, Y, theta, counts)  # 得到损失值
    ax.plot(np.arange(counts), costs, label=alpha)  # 设置x轴参数为迭代次数，y轴参数为cost
    ax.legend()  # 显示label

ax.set(xlabel='counts',  # 图的坐标轴设置
       ylabel='cost',
       title='cost vs counts')  # 标题
plt.show()  # 显示图像