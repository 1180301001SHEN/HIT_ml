import pandas as pd
import numpy as np
import itertools


class IrisDataSet(object):
    def __init__(self):
        self.x = np.array(pd.read_excel("./iris.xlsx", usecols=[0, 1, 2, 3]))
        self.y = np.array(pd.read_excel("./iris.xlsx", usecols=[4]))
        # 由于我们不知道clustering分完类后哪一个标签对应哪一个标签，所以我们直接做一个全排列
        self.permutateClass = list(itertools.permutations(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 3))

    def getData(self):
        return self.x

    def accuracy(self, label):
        number = len(label)
        counts = []
        for i in range(len(self.permutateClass)):
            count = 0
            for j in range(number):
                # 由于我们并不知道分成的类别顺序，我们对Aij个组合都计算一遍
                if self.y[j] == self.permutateClass[i][label[j]]:
                    count += 1
            counts.append(count)
        return np.max(counts) * 1.0 / number
