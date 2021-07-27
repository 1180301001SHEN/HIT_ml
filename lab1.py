# by 1180301001 沈汝佳
import numpy as np
import math
import matplotlib.pyplot as plt


# 生成数据
def generateNumbers(size, range):
    x0 = np.linspace(0, range, size).reshape(size, 1)  # 均匀分布
    x0 = np.sort(x0, 0)
    mistake = np.random.randn(size, 1)*0.1  # 高斯分布
    Y = np.sin(x0)+mistake                  # 加入噪声
    return np.matrix(x0), np.matrix(Y)


# 生成1,x0,x0^2,...,x0^(m-1)
def ModelGenerateMatrix(x0, m, size):
    x = np.ones((size, 1))
    for i in range(1, m):
        x = np.hstack((x, np.power(x0, i)))     # size*m大小的一个矩阵
    return np.matrix(x)


# 解析解
def AnalyticSolution(X, T):
    w = np.dot(np.dot(X.T, X).I, np.dot(X.T, T))               # w=(X'X)^{-1}X'T
    return w


# 解析解（带惩罚项）
def AnalyticSolutionWithLambda(X, T, m, lambd):
    w = np.dot((np.dot(X.T, X) + np.dot(lambd, np.eye(m))).I, np.dot(X.T, T))   # w=(X'X+lambda I)^{-1}X'T
    return w


# 梯度函数(使用均方误差代价函数)
def GradientDescentFunction(x, y, w, size):
    return np.dot(x.T, (np.dot(x, w) - y)) * (1.0/size)       # deta=(X'Xw-X'Y)/size


# 损失函数(使用均方误差代价函数)
def LossFunction(x, y, w, size):
    subs = np.dot(x, w) - y
    loss = np.dot(subs.T, subs)/(2.0 * size)                   # loss=||Xw-y||_2/2size
    return loss[0, 0]


# 梯度下降(使用均方误差代价函数)
def GradientDescent(x, y, learning_rate, m, size):
    w1 = np.zeros((m, 1))
    gradient = GradientDescentFunction(x, y, w1, size)
    print("gradient=", gradient)
    epsilon = 1e-3
    lr = learning_rate
    while np.linalg.norm(gradient) > epsilon:           # 这里面一定要动态改变学习率，否则直接梯度上升
        w2 = w1 - lr * gradient
        gradientPre = gradient
        gradient = GradientDescentFunction(x, y, w2, size)
        while np.linalg.norm(gradient) > np.linalg.norm(gradientPre):
            lr = lr / 2
            w2 = w1 - lr * gradient
            gradient = GradientDescentFunction(x, y, w2, size)
            print(np.linalg.norm(gradient))
            print(1)
        w1 = w2
        print(np.linalg.norm(gradient))
    return w1


# 梯度函数,带惩罚项
def GradientDescentFunctionWithLambda(x, y, w, size, lambd):
    return (np.dot(np.dot(x.T, x), w) - np.dot(x.T, y) + lambd * w)*(1.0/size)   # deta=(X'Xw-X'Y+lambda w)/size


# 损失函数(使用均方误差代价函数)
def LossFunctionWithLambda(x, y, w, size, lambd):
    subs = np.dot(x, w) - y
    loss = (np.dot(subs.T, subs)/(2.0 * size)) + (lambd/2) * np.dot(w.T, w)                   # loss=||Xw-y||_2/2size
    return loss[0, 0]


# 梯度下降,带惩罚项(使用均方误差代价函数)
def GradientDescentWithLambda(x, y, learning_rate, m, size, lambd):
    w1 = np.zeros((m, 1))
    gradient = GradientDescentFunctionWithLambda(x, y, w1, size, lambd)
    print("gradient=", gradient)
    epsilon = 1e-3
    lr = learning_rate
    while np.linalg.norm(gradient) > epsilon:               # 这里面一定要动态改变学习率，否则直接梯度上升
        w2 = w1 - lr * gradient
        gradientPre = gradient
        gradient = GradientDescentFunctionWithLambda(x, y, w2, size, lambd)
        while np.linalg.norm(gradient) > np.linalg.norm(gradientPre):
            lr = lr / 2
            w2 = w1 - lr * gradient
            gradient = GradientDescentFunctionWithLambda(x, y, w2, size, lambd)
            print(np.linalg.norm(gradient))
            print(1)
        w1 = w2
        print(np.linalg.norm(gradient))
    return w1


# 共轭梯度下降法(使用均方误差代价函数)
def ConjugatedGradient(x, y, size, m):
    A = (1/size) * np.dot(x.T, x)
    w = np.zeros((m, 1))
    r = GradientDescentFunction(x, y, w, size)
    p = r
    for i in range(1, size):                            # 求一组正交基
        alpha = float(np.dot(r.T, p)/np.dot(np.dot(p.T, A), p))
        w = w + alpha * p
        r = r - alpha * np.dot(A, p)
        beta = -float(np.dot(r.T, np.dot(A, p))/np.dot(p.T, np.dot(A, p)))
        p = r + beta * p
    return -w


# 共轭梯度下降法(使用均方误差代价函数)
def ConjugatedGradientWithLambda(x, y, size, m, lambd):
    A = (1/size) * np.dot(x.T, x) + lambd * np.eye(m)
    w = np.zeros((m, 1))
    r = GradientDescentFunctionWithLambda(x, y, w, size, lambd)
    p = r
    for i in range(1, size):                            # 求一组正交基
        alpha = float(np.dot(r.T, p)/np.dot(np.dot(p.T, A), p))
        w = w + alpha * p
        r = r - alpha * np.dot(A, p)
        beta = -float(np.dot(r.T, np.dot(A, p))/np.dot(p.T, np.dot(A, p)))
        p = r + beta * p
    return -w


def TestAnalyticSolution(size, numrange, lambd, m):
    x0, y = generateNumbers(size, numrange)
    print(x0)
    print("...")
    print(y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=2, ymin=-2)
    plt.scatter(x0.tolist(), y.tolist())
    x = ModelGenerateMatrix(x0, m, size)
    print(x)
    # 解析解测试
    w0 = AnalyticSolution(x, y)
    print("w0=", w0)
    xw0 = np.random.random((100, 1)) * numrange
    xw0 = np.sort(xw0, 0)
    xw0_0 = ModelGenerateMatrix(xw0, m, 100)
    yw0 = np.dot(xw0_0, w0)
    print(yw0)
    plt.plot(xw0.tolist(), yw0.tolist(), label='AnalyticSolution')
    w1 = AnalyticSolutionWithLambda(x, y, m, lambd)
    print("w1=", w1)
    yw1 = np.dot(xw0_0, w1)
    plt.plot(xw0.tolist(), yw1.tolist(), label='AnalyticSolutionWithLambda')
    x_sin = np.linspace(0, math.pi*2, 100).reshape(100, 1)
    y_sin = np.sin(x_sin)
    plt.plot(x_sin, y_sin, label='y=sinx')
    plt.legend()
    plt.show()


def TestGradientDescent(size, numrange, lambd, m, learning_rate):
    x0, y = generateNumbers(size, numrange)
    print(x0)
    print("...")
    print(y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=2, ymin=-2)
    plt.scatter(x0.tolist(), y.tolist())
    x = ModelGenerateMatrix(x0, m, size)
    print(x)
    # 梯度下降测试
    w2 = GradientDescent(x, y, learning_rate, m, size)
    print("w2=", w2)
    w3 = GradientDescentWithLambda(x, y, learning_rate, m, size, lambd)
    print("w3=", w3)
    xw2 = np.random.random((100, 1)) * numrange
    xw2 = np.sort(xw2, 0)
    xw2_2 = ModelGenerateMatrix(xw2, m, 100)
    yw2 = np.dot(xw2_2, w2)
    print(yw2)
    plt.plot(xw2.tolist(), yw2.tolist(), label='GradientDescent')
    yw3 = np.dot(xw2_2, w3)
    plt.plot(xw2.tolist(), yw3.tolist(), label='GradientDescentWithLambda')
    x_sin = np.linspace(0, math.pi*2, 100).reshape(100, 1)
    y_sin = np.sin(x_sin)
    plt.plot(x_sin, y_sin, label='y=sinx')
    plt.legend()
    plt.show()


def TestConjugatedGradient(size, numrange, lambd, m, learning_rate):
    x0, y = generateNumbers(size, numrange)
    print(x0)
    print("...")
    print(y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=2, ymin=-2)
    plt.scatter(x0.tolist(), y.tolist())
    x = ModelGenerateMatrix(x0, m, size)
    print(x)
    # 共轭梯度下降测试
    w4 = ConjugatedGradient(x, y, size, m)
    print("w4=", w4)
    xw4 = np.random.random((100, 1)) * numrange
    xw4 = np.sort(xw4, 0)
    xw4_4 = ModelGenerateMatrix(xw4, m, 100)
    yw4 = np.dot(xw4_4, w4)
    print(yw4)
    plt.plot(xw4.tolist(), yw4.tolist(), label='ConjugatedGradient')
    w5 = ConjugatedGradientWithLambda(x, y, size, m, lambd)
    print("w5=", w5)
    yw5 = np.dot(xw4_4, w5)
    plt.plot(xw4.tolist(), yw5.tolist(), label='ConjugatedGradientWithLambda')
    x_sin = np.linspace(0, math.pi*2, 100).reshape(100, 1)
    y_sin = np.sin(x_sin)
    plt.plot(x_sin, y_sin, label='y=sinx')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 测试y=sinx+Gaussion
    # x, y = generateNumbers(100, math.pi*2)
    # plt.plot(x, y, label='y=sinx+Gaussion')
    # x2 = np.linspace(0, math.pi*2, 100).reshape(100, 1)
    # y2 = np.sin(x2)
    # plt.plot(x2, y2, label='y=sinx')
    # plt.legend()
    # plt.show()

    # # 解析解法
    # # 解析解 size=6,numrange=pi*2,lambd=0.01,m=6
    # TestAnalyticSolution(6, math.pi*2, 0.01, 6)
    # # 解析解 size=9,numrange=pi*2,lambd=0.01,m=6
    # TestAnalyticSolution(9, math.pi*2, 0.01, 6)
    # # 解析解 size=20,numrange=pi*2,lambd=0.01,m=6
    # TestAnalyticSolution(20, math.pi*2, 0.01, 6)

    # # 解析解 size=6,numrange=pi*2,lambd=0.01,m=9
    # TestAnalyticSolution(6, math.pi*2, 0.01, 9)
    # # 解析解 size=9,numrange=pi*2,lambd=0.01,m=9
    # TestAnalyticSolution(9, math.pi*2, 0.01, 9)
    # # 解析解 size=20,numrange=pi*2,lambd=0.01,m=9
    # TestAnalyticSolution(20, math.pi*2, 0.01, 9)

    # # 解析解 size=6,numrange=pi*2,lambd=0.1,m=9
    # TestAnalyticSolution(6, math.pi*2, 0.1, 9)
    # # 解析解 size=9,numrange=pi*2,lambd=0.1,m=9
    # TestAnalyticSolution(9, math.pi*2, 0.1, 9)
    # # 解析解 size=20,numrange=pi*2,lambd=0.1,m=9
    # TestAnalyticSolution(20, math.pi*2, 0.1, 9)

    # # 共轭梯度法
    # # 共轭解 size=6,numrange=pi*2,lambd=0.01,m=6,learing_rate=0.001
    # TestConjugatedGradient(6, math.pi*2, 0.01, 6, 0.001)
    # # 共轭解 size=9,numrange=pi*2,lambd=0.01,m=6,learing_rate=0.001
    # TestConjugatedGradient(9, math.pi*2, 0.01, 6, 0.001)
    # # 共轭解 size=20,numrange=pi*2,lambd=0.01,m=6,learing_rate=0.001
    # TestConjugatedGradient(20, math.pi*2, 0.01, 6, 0.001)

    # # 共轭解 size=6,numrange=pi*2,lambd=0.01,m=9,learing_rate=0.001
    # TestConjugatedGradient(6, math.pi*2, 0.01, 9, 0.001)
    # # 共轭解 size=9,numrange=pi*2,lambd=0.01,m=9,learing_rate=0.001
    # TestConjugatedGradient(9, math.pi*2, 0.01, 9, 0.001)
    # # 共轭解 size=20,numrange=pi*2,lambd=0.01,m=9,learing_rate=0.001
    # TestConjugatedGradient(20, math.pi*2, 0.01, 9, 0.001)

    # # 共轭解 size=6,numrange=pi*2,lambd=0.1,m=9,learing_rate=0.001
    # TestConjugatedGradient(6, math.pi*2, 0.1, 9, 0.001)
    # # 共轭解 size=9,numrange=pi*2,lambd=0.1,m=9,learing_rate=0.001
    # TestConjugatedGradient(9, math.pi*2, 0.1, 9, 0.001)
    # # 共轭解 size=20,numrange=pi*2,lambd=0.1,m=9,learing_rate=0.001
    # TestConjugatedGradient(20, math.pi*2, 0.1, 9, 0.001)

    # # 梯度下降法
    # # 梯度解 size=3,numrange=pi*2,lambd=0.01,m=1,learing_rate=0.001
    # TestGradientDescent(3, math.pi*2, 0.01, 1, 0.001)
    # # 梯度解 size=3,numrange=pi*2,lambd=0.01,m=2,learing_rate=0.001
    # TestGradientDescent(3, math.pi*2, 0.01, 2, 0.001)
    # # 梯度解 size=3,numrange=pi*2,lambd=0.01,m=3,learing_rate=0.001
    # TestGradientDescent(3, math.pi*2, 0.01, 3, 0.001)

    # # 梯度解 size=10,numrange=pi*2,lambd=0.01,m=1,learing_rate=0.001
    # TestGradientDescent(10, math.pi*2, 0.01, 1, 0.001)
    # # 梯度解 size=10,numrange=pi*2,lambd=0.01,m=2,learing_rate=0.001
    # TestGradientDescent(10, math.pi*2, 0.01, 2, 0.001)
    # # 梯度解 size=10,numrange=pi*2,lambd=0.01,m=3,learing_rate=0.001
    # TestGradientDescent(10, math.pi*2, 0.01, 3, 0.001)

    # # 梯度解 size=20,numrange=pi*2,lambd=0.01,m=1,learing_rate=0.001
    # TestGradientDescent(20, math.pi*2, 0.01, 1, 0.001)
    # # 梯度解 size=20,numrange=pi*2,lambd=0.01,m=2,learing_rate=0.001
    # TestGradientDescent(20, math.pi*2, 0.01, 2, 0.001)
    # # 梯度解 size=20,numrange=pi*2,lambd=0.01,m=3,learing_rate=0.001
    # TestGradientDescent(20, math.pi*2, 0.01, 3, 0.001)

    # 梯度解 size=10,numrange=pi*2,lambd=0.01,m=4,learing_rate=0.001
    TestGradientDescent(10, math.pi*2, 0.01, 4, 0.001)
    # 梯度解 size=10,numrange=pi*2,lambd=0.01,m=4,learing_rate=0.001
    # TestGradientDescent(100, math.pi*2, 0.01, 4, 0.001)
