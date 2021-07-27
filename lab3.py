import numpy as np
import matplotlib.pyplot as plt
import K_MEANS
import GMM
import ReadIris


def GenerateData(kthMean, kthNumber, k):
    cov = [[0.5, 0], [0, 0.5]]
    data = []
    # 生成K类数据
    for i in range(k):
        for j in range(kthNumber[i]):
            data.append(np.random.multivariate_normal(kthMean[i], cov).tolist())
    return np.array(data)


if __name__ == "__main__":
    k = 4
    kthMean = [[0, 0], [1, 2], [-5, -5], [2, 3]]
    kthNumber = [50, 50, 50, 50]
    data = GenerateData(kthMean, kthNumber, k)
    print(data)
    km = K_MEANS.KMEANS(data, k)
    RandomMU, RandomLabel = km.KmeansChooseRandom()
    NormalMu, NormalLabel = km.KmeansChooseNotRandom()

    # KMEANS画随机选择点时的图
    plt.subplot(2, 2, 1)
    plt.title("KMeans:randomly")
    for i in range(k):
        plt.scatter(np.array(RandomLabel[i])[:, 0], np.array(RandomLabel[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(RandomMU[:, 0], RandomMU[:, 1], edgecolors="r", label="center")
    plt.legend()

    # KMEANS画最远距离选取中心点时的图
    plt.subplot(2, 2, 2)
    plt.title("KMeans:MaxDistance")
    for i in range(k):
        plt.scatter(np.array(NormalLabel[i])[:, 0], np.array(NormalLabel[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(NormalMu[:, 0], NormalMu[:, 1], edgecolors="r", label="center")
    plt.legend()

    # GMM
    gmm = GMM.GMM(data, k)
    GMMmu, GMMlabel = gmm.RunGMM()
    plt.subplot(2, 2, 3)
    plt.title("GMM")
    for i in range(k):
        plt.scatter(np.array(GMMlabel[i])[:, 0], np.array(GMMlabel[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(GMMmu[:, 0], GMMmu[:, 1], edgecolors="r", label="center")
    plt.legend()

    iris = ReadIris.IrisDataSet()
    irisData = iris.getData()
    # iris+KMEANS
    KMEANSIris = K_MEANS.KMEANS(irisData, 3)
    KmeansIrisMu, KmeansIrisLabel = KMEANSIris.KmeansChooseNotRandom()
    print("IrisWithKMeans:", iris.accuracy(KMEANSIris.sampleLabel))
    # iris+GMM
    GMMIris = GMM.GMM(irisData, 3)
    GMMIrisMu, GMMIrisLabel = GMMIris.RunGMM()
    print("IrisWithGMM:", iris.accuracy(GMMIris.sampleLabel))

    plt.show()
