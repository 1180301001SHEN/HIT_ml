import numpy as np
import math
import matplotlib.pyplot as plt


# 画出散点图和判别函数
def DrawScatterPlot(x, y, DiscriminantFunction):
    # 以x0,x1为坐标轴,颜色是y的值
    plt.scatter(x[:, 0], x[:, 1], c=y)
    # 如果输入了判别函数,那就画出来
    if DiscriminantFunction:
        x1 = min(x[:, 0])+(max(x[:, 0]) - min(x[:, 0])) * np.random.random(50)
        x2 = DiscriminantFunction(x1)
        plt.plot(x1, x2, 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('ScatterPlot')
    plt.show()


# number是生成样本的数量，每个样本有两维，independence是样本之间是否独立
def MyGenerateDate(number, independence):
    # 一半正例一半反例
    n = math.ceil(number/2)
    # x是一个number*2的矩阵,y是一个number*1的向量
    x = np.zeros((number, 2))
    y = np.zeros(number)
    # x正反例的均值
    mean1 = [1, 2]
    mean2 = [-1, -2]
    # xi和xj独立时的协方差矩阵
    cov_independence = [[0.5, 0], [0, 0.5]]
    # xi和xj不独立时的协方差矩阵
    cov_dependence = [[1, 0.5], [0.5, 1]]
    if independence:
        x[:n, :] = np.random.multivariate_normal(mean1, cov_independence, size=n)
        x[n:, :] = np.random.multivariate_normal(mean2, cov_independence, size=number-n)
        y[:n] = 0
        y[n:] = 1
    else:
        x[:n, :] = np.random.multivariate_normal(mean1, cov_dependence, size=n)
        x[n:, :] = np.random.multivariate_normal(mean2, cov_dependence, size=number-n)
        y[:n] = 0
        y[n:] = 1
    # 由于画图的时候不能用matrix,因此这里没有将array转成matrix
    return x, y


# 判别函数(假定这个是真实的)
def DiscriminantFunction(x):
    return -1/2 * x + 0.5


# 类Sigmoid函数
def Sigmoid(b):
    return 1/(1 + np.exp(b))


# 计算似然函数
def LikelihoodFunction(x, y, w):
    size = np.size(x, axis=0)
    temp = np.zeros((size, 1))
    for i in range(size):
        temp[i] = np.dot(w, x[i].T)
    sumln = 0
    for i in range(size):
        sumln += np.log(1 + np.exp(temp[i]))
    return np.dot(y, temp) - sumln


# 梯度下降(带惩罚项,当lambda赋值成0就是不带惩罚项的了)
def GradientDescentWithLambda(x, y, epoch, eta, eps, dimension, lamb):
    size = np.size(x, axis=0)
    w = np.ones((1, dimension + 1))
    # loss和epch为了画损失函数随迭代次数的图
    loss = np.zeros(epoch)
    epch = np.zeros(epoch)
    # 最大的迭代次数
    for i in range(epoch):
        loss1 = -1/size * LikelihoodFunction(x, y, w)
        # 计算Xw^T
        temp = np.zeros((size, 1))
        for j in range(size):
            temp[j] = np.dot(w, x[j].T)
        # 计算梯度g=Y^T-sum()
        gradint = np.dot(y, x)
        for k in range(size):
            gradint -= Sigmoid(temp[k]) * np.exp(temp[k]) * x[k]
        w = w - eta * lamb * w + eta * (1/size) * gradint
        loss2 = -1/size * LikelihoodFunction(x, y, w)
        epch[i] = i
        loss[i] = loss2
        if i % 100 == 0:
            print(i, 'loss=', loss2, ',w=', w, 'gradient=', gradint)
        # 如果两次loss相差不大,就认为是收敛了
        if abs(loss2 - loss1) < eps:
            epch = epch[:i+1]
            loss = loss[:i+1]
            break
    return w, epch, loss


# 构造
def LinearRegression(number, lamb, independence):
    # 生成初始的x,y
    x, y = MyGenerateDate(number, independence)
    # DrawScatterPlot(x, y, DiscriminantFunction)
    # 构造x使其符合number*3
    X = np.ones((number, 3))
    X[:, 1] = x[:, 0]
    X[:, 2] = x[:, 1]
    # 参数,epoch训练次数,eta学习率,eps精度
    epoch = 1000000
    eta = 0.1
    eps = 1e-4
    # 训练
    w, epch, loss = GradientDescentWithLambda(X, y, epoch, eta, eps, np.size(x, axis=1), lamb)
    # 将w转换成向量
    w = w.reshape(3)
    # 将ax+by+z=0改写成y=ax+b的形式
    coefficient = -(w / w[2])[0:2]
    discriminant = np.poly1d(coefficient[::-1])
    print('The predict discriminant function: y = ')
    print(discriminant)
    DrawScatterPlot(x, y, discriminant)
    plt.plot(epch, loss, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    return w


# 测试正确率
def TestMyGenerateCorrect(w, number, independence):
    x, y = MyGenerateDate(number, independence)
    X = np.ones((number, 3))
    X[:, 1] = x[:, 0]
    X[:, 2] = x[:, 1]
    correct = 0
    for i in range(number):
        label = 0
        if np.dot(w, X[i].T) >= 0:
            label = 1
        else:
            label = 0
        if label == y[i]:
            correct += 1
    rate = correct / number
    print("正确率：", rate)


# 选取的是iris数据集,但是由于iris是一个多分类问题,
# 我们在逻辑回归中只考虑而分类问题,这里我就去掉了一种类别
# 如果类别是Iris-setosa就标记成1
# 如果是Iris-versicolor就标记成0

# 读数据,构造x,y矩阵
def readIrisData():
    f = open('./iris.txt', 'r')
    k = 0
    x = np.ones((100, 5))
    y = np.ones((100, 1))
    # 鸢尾花数据集每一行有4个特征和一个类别,中间用','分隔
    for i in f:
        i = i.strip()
        temp = i.split(',')
        x[k][1] = float(temp[0])
        x[k][2] = float(temp[1])
        x[k][3] = float(temp[2])
        x[k][4] = float(temp[3])
        if temp[4] == 'Iris-versicolor':
            y[k][0] = 0
        else:
            y[k][0] = 1
        k += 1
    # 分割训练集和测试集,比例4:1
    train_x = np.zeros((80, 5))
    train_y = np.zeros((80, 1))
    test_x = np.zeros((20, 5))
    test_y = np.zeros((20, 1))
    count = 0
    for i in range(100):
        if i % 5 == 0:
            k = int(i/5)
            test_x[k] = x[i]
            test_y[k] = y[i]
        else:
            train_x[count] = x[i]
            train_y[count] = y[i]
            count += 1
    return train_x, train_y.T, test_x, test_y.T


# 构造对iris数据集的逻辑回归
def IrisLinearRegression(lamb):
    train_x, train_y, test_x, test_y = readIrisData()
    # 参数,epoch训练次数,eta学习率,eps精度
    epoch = 100000
    eta = 0.1
    eps = 1e-4
    dimension = np.size(train_x, axis=1)
    # 训练
    w, epch, loss = GradientDescentWithLambda(train_x, train_y, epoch, eta, eps, dimension-1, lamb)
    # 将w转换成向量
    w = w.reshape(dimension)
    # 将w0+w1x1+w2x2+w3x3+w4x4+w5x5=0改写成x5=w0'+w1'x1+w2'x2+w3'x3+w4'x4的形式
    coefficient = -(w / w[dimension-1])[0:dimension-1]
    discriminant = np.poly1d(coefficient[::-1])
    print('The predict discriminant function: y = ')
    print(discriminant)
    plt.plot(epch, loss, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    correct = 0
    for i in range(20):
        label = 0
        if np.dot(w, test_x[i].T) >= 0:
            label = 1
        else:
            label = 0
        if label == test_y[0][i]:
            correct += 1
    rate = correct / 20
    print("正确率：", rate)


if __name__ == "__main__":
    # x, y = MyGenerateDate(100, True)
    # DrawScatterPlot(x, y, DiscriminantFunction)

    # w = LinearRegression(1000, 0.2, True)
    # TestMyGenerateCorrect(w, 10000, True)

    # w = LinearRegression(1000, 0.2, False)
    # TestMyGenerateCorrect(w, 10000, False)

    IrisLinearRegression(0.5)
