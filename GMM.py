import numpy as np
from scipy.stats import multivariate_normal
import collections


# 初始化中mu是k*m维,sigma是n*n*k维,likelihood是n*k维,alpha是1*k维
class GMM(object):
    def __init__(self, data, k, delta=1e-6, iteration=10000):
        self.data = data
        self.k = k
        self.delta = delta
        self.rows, self.col = self.data.shape
        self.iteration = iteration
        self.alpha = np.ones(self.k) * (1.0/self.k)
        self.mu, self.sigma = self.InitMuSigma()
        self.sampleLabel = None
        self.label = collections.defaultdict(list)
        self.PM = None
        self.lastAlpha = self.alpha
        self.lastMu = self.mu
        self.lastSigma = self.sigma

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

    # 初始化mu和Sigma
    def InitMuSigma(self):
        mu = self.InitialCenterNotRandom()
        # 对于每一个类，都有一个n*n的协方差矩阵，所以Sigma应该是k*n*n的
        sigma = collections.defaultdict(list)
        for i in range(self.k):
            sigma[i] = np.eye(self.col, dtype=float)
        return mu, sigma

    def LikeliHood(self):
        # 这里面likelihood每一列对应一个类别高斯分布的概率
        likelihood = np.zeros((self.rows, self.k))
        for i in range(self.k):
            likelihood[:, i] = multivariate_normal.pdf(self.data, self.mu[i], self.sigma[i])
        return likelihood

    def EStep(self):
        # 应该是alpha[i]*likelihood[j][i],likelihood的第j行乘以alpha第i列
        WeightLL = self.LikeliHood() * self.alpha
        # WeightLL是n*k维,SumLL应该对每一行求和,得到n*1维矩阵
        SumLL = np.expand_dims(np.sum(WeightLL, axis=1), axis=1)
        # PM的每一个是P(z_j=i|x_j)
        self.PM = WeightLL / SumLL
        # 对PM的每一行找最大,记录下标
        self.sampleLabel = self.PM.argmax(axis=1)
        for i in range(self.rows):
            self.label[self.sampleLabel[i]].append(self.data[i].tolist())

    def MStep(self):
        for i in range(self.k):
            # 取出所有的P(z=i|x)
            PM = np.expand_dims(self.PM[:, i], axis=1)
            # 将data和PM的对应位相乘得到sum(P(z_j=i|x_j)x_j),mu_i是1*m的
            mu_i = (PM * self.data).sum(axis=0)/PM.sum()
            cov_i = (self.data - mu_i).T.dot((self.data - mu_i) * PM)/PM.sum()
            self.mu[i], self.sigma[i] = mu_i, cov_i
        self.alpha = self.PM.sum(axis=0)/self.rows

    def Stop(self):
        # 这里的loss是alpha,sigma,mu的距离和
        loss = np.linalg.norm(self.lastAlpha - self.alpha)
        loss = loss + np.linalg.norm(self.lastMu - self.mu)
        loss = loss + np.sum([np.linalg.norm(self.lastSigma[i] - self.sigma[i]) for i in range(self.k)])
        if loss > self.delta:
            self.lastAlpha = self.alpha
            self.lastMu = self.mu
            self.lastSigma = self.sigma
            return False
        else:
            return True

    def RunGMM(self):
        # 交替进行E-Step和M-Step
        for i in range(self.iteration):
            self.EStep()
            self.MStep()
            if self.Stop():
                break
        self.EStep()
        return self.mu, self.label
