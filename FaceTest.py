from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
import myPCA


def SNR(source, target):
    # 计算两张图片的偏差
    diff = source - target
    cols = len(source)
    # 计算原始图片的方差
    sourceMean = (1 / cols) * np.sum(source)
    sourceCov = (1/cols) * np.sum([(source[i]-sourceMean) ** 2 for i in range(cols)])
    # 计算偏差的方差
    diffMean = (1 / cols) * np.sum(diff)
    diffCov = (1/cols) * np.sum([(diff[i]-diffMean) ** 2 for i in range(cols)])
    m = sourceCov / diffCov
    # 将单位转换成dB
    return 10 * np.log10(m)


def SNRGraph(origindata, PCAdata):
    rows, cols = origindata.shape
    # 对每一张图片求一个SNR,然后做平均
    temp = np.abs(np.mean([SNR(origindata[i, :], PCAdata[i, :]) for i in range(rows)]))
    print("信噪比:", temp, "dB")


if __name__ == "__main__":
    faces = fetch_lfw_people(min_faces_per_person=60)
    # 一共有1349张图像，数据特征2914个
    print(faces.data.shape)
    # 一共1349张图像，每个图像是62*47=2914的
    print(faces.images.shape)

    x = faces.data
    X = x.T
    print(X.shape)
    p = myPCA.myPCA(X, 50)
    xpca, featureVectors, xmean, q = p.reduceDimension()
    print(xpca.shape)
    print("q:", q.shape)

    fig, axes = plt.subplots(4, 8, figsize=[10, 5], subplot_kw={"xticks": [], "yticks": []})
    # 绘制子图4行8列 设置坐标轴为空
    for i, ax in enumerate(axes.flat):
        # 展平为一维准备绘制图像进子图
        ax.imshow(faces.data[i, :].reshape(62, 47), cmap='gray')
        # 绘制子图，设置颜色模式为灰色

    xPCA = np.array((np.dot(np.dot(q, q.T), (X - xmean)) + xmean).T, dtype=int)
    print(xPCA.shape)
    fig, axes = plt.subplots(4, 8, figsize=[10, 5], subplot_kw={"xticks": [], "yticks": []})
    # 绘制子图4行8列 设置坐标轴为空
    for i, ax in enumerate(axes.flat):
        # 展平为一维准备绘制图像进子图
        ax.imshow(xPCA[i, :].reshape(62, 47), cmap='gray')
        # 绘制子图，设置颜色模式为灰色

    plt.show()
    SNRGraph(x, xPCA)
