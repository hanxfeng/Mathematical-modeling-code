import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def kmean(X,n):
    """
    :param X: 可以是 Pandas DataFrame 或 NumPy 数组，每列代表一个特征（维度）
    :param n: 指定聚类的簇数量
    :return:无返回值，但会生成聚类可视化图表，并将聚类结果保存到 Excel 文件
    数据需要按列排序，每列是一维数据
    """

    # 进行kmeans聚类
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # 获取簇的中心点
    centroids = kmeans.cluster_centers_

    # 可视化结果
    X_list = X.columns
    print(X.shape[0])
    rows, columns = X.shape
    if columns==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap('viridis', n)
        for i in range(X.shape[0]):
            ax.scatter(X.iloc[i, 0], X.iloc[i, 1], X.iloc[i, 2], c=colors[y_kmeans[i]], s=50, alpha=0.6)

        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=200, alpha=0.75,
                   marker='^')

        ax.set_xlabel(f'{X_list[0]}')
        ax.set_ylabel(f'{X_list[1]}')
        ax.set_zlabel(f'{X_list[2]}')
        plt.show()

    elif columns == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
        plt.xlabel(f'{X_list[0]}')
        plt.ylabel(f'{X_list[1]}')
        plt.show()
    else:
        print("数据维数大于3无法绘图")

    #输出聚类结果
    y_kmeans = y_kmeans.T
    y_kmeans = pd.DataFrame(y_kmeans)
    X = pd.concat([X, y_kmeans], axis=1)
    X.to_excel('kmeans聚类数据.xlsx')

