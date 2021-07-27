import numpy as np
import collections
import random


class KMEANS(object):
    def __init__(self, data, k, delta=1e-6):
        self.data = data
        self.k = k
        self.delta = delta
        self.rows, self.col = data.shape
        self.mu = self.InitialCenterNotRandom()
        self.sampleLabel = [-1] * self.rows

    @staticmethod
    def EuclideanDistance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def InitialCenterNotRandom(self):
        # 随机选择第一个初始的点
        mu0 = np.random.randint(0, self.rows)
        mu = [self.data[mu0]]
        # 选择与当前mu最远的点作为中心点
        for i in range(self.k - 1):
            temp = []
            for j in range(self.rows):
                # 求出data[j]和每一个mu的距离之和
                temp.append(np.sum([self.EuclideanDistance(self.data[j], mu[k]) for k in range(len(mu))]))
            # 选择求和最大的那个就是下一个mu[i]
            mu.append(self.data[np.argmax(temp)])
        return np.array(mu)

    def Kmeans(self):
        while True:
            # 用字典记录每一个mu对应的data
            label = collections.defaultdict(list)
            for i in range(self.rows):
                # data[i]和所有mu的距离
                dij = [self.EuclideanDistance(self.data[i], self.mu[j]) for j in range(self.k)]
                # 找出data[i]应该属于的mu
                lambdai = np.argmin(dij)
                # 将其放在(mu,[data])这个字典里
                label[lambdai].append(self.data[i].tolist())
                self.sampleLabel[i] = lambdai

            # 更新mu(利用每个mu对应的data的均值作为新的mu)
            NewMu = np.array([np.mean(label[i], axis=0).tolist() for i in range(self.k)])
            # 计算mu的差距，直到两次迭代相差无几
            DeltaMu = np.sum(self.EuclideanDistance(self.mu[i], NewMu[i]) for i in range(self.k))
            if DeltaMu > self.delta:
                self.mu = NewMu
            else:
                break

            print("mu:", self.mu)
        return self.mu, label

    def KmeansChooseRandom(self):
        self.mu = self.data[random.sample(range(self.rows), self.k)]
        return self.Kmeans()

    def KmeansChooseNotRandom(self):
        self.mu = self.InitialCenterNotRandom()
        return self.Kmeans()
