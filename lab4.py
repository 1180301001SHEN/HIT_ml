import numpy as np
import myPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generateData(dataDimension, number=100):
    # 生成2维数据
    if dataDimension == 2:
        mean = [-2, 2]
        cov = [[1, 0], [0, 1]]
    # 生成3维数据
    elif dataDimension == 3:
        mean = [0, 2, 4]
        cov = [[0.001, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        assert False
    sample = []
    # 生成一个样本数量为number,均值方差由上面给定的样本矩阵
    for i in range(number):
        sample.append(np.random.multivariate_normal(mean, cov).tolist())
    return np.array(sample).T


def draw(dimension, data, PCAData):
    tempdata = np.mat(data).T
    tempPCAData = np.mat(PCAData).T
    if dimension == 2:
        plt.scatter(tempdata[:, 0].tolist(), tempdata[:, 1].tolist(), facecolor="none", edgecolor="b", label="Origin Data")
        plt.scatter(tempPCAData[:, 0].tolist(), tempPCAData[:, 1].tolist(), facecolor='r', label='PCA Data')
    elif dimension == 3:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(data[0, :].tolist(), data[1, :].tolist(), data[2, :].tolist(), c='b', label='Origin Data')
        ax.scatter3D(PCAData[0, :].tolist(), PCAData[1, :].tolist(), PCAData[2, :].tolist(), c='r', label='PCA Data')
    else:
        assert False
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dimension = 3
    number = 100
    x = generateData(dimension, number)
    print(x)
    print(x.shape)
    p = myPCA.myPCA(x, dimension-1)
    xpca, featureVectors, xmean, q = p.reduceDimension()
    print(xpca.shape)
    print("Feature vectors:", q)

    pcaData = np.dot(np.dot(q, q.T), (x - xmean)) + xmean
    draw(dimension, x, pcaData)
