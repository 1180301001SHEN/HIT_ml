import numpy as np
import pandas as pd


class myPCA(object):
    def __init__(self, x, dimension):
        # 假设x是D*N维的数据，有n个样本点每个样本点是d维的
        self.x = x
        # dimension是想要降到的维度
        self.dimension = dimension
        self.rows, self.col = x.shape

    def cov(self):
        # 计算均值d*1维向量
        xmean = np.sum(self.x, axis=1).reshape((self.rows, 1))
        allone = np.ones((1, self.col))
        xmeanTo = np.dot(xmean, allone)
        # 计算协方差矩阵(1/N)(X-xmean)(X-xmean)^T
        covx = (1 / self.col) * np.dot(self.x - xmeanTo, (self.x - xmeanTo).T)
        return covx, xmeanTo

    def getFeature(self):
        covx, xmeanTo = self.cov()
        # 解析出特征值和特征向量
        eigenvalues, featureVectors = np.linalg.eig(covx)
        m = eigenvalues.shape[0]
        # 将特征值和特征向量合并排列
        temp = np.vstack((eigenvalues.reshape((1, m)), featureVectors))
        tempDF = pd.DataFrame(temp)
        tempSort = tempDF.sort_values(0, ascending=False, axis=1)
        return tempSort, featureVectors, xmeanTo

    def reduceDimension(self):
        tempSort, featureVectors, xmeanTo = self.getFeature()
        # 返回前dimension大的特征值对应的特征向量
        p = tempSort.values[1:, 0:self.dimension]
        y = np.dot(p.T, self.x)
        return y, featureVectors, xmeanTo, p
