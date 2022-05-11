import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from stock_data import get_list


def k_means(k):
    """

    data: All the data points in "data" which corespond to a label that is equal to 0,...
    will be put into cluster 1, ...
        cluster_1 = data [label == 0]
        cluster_2 = data[label == 1]
        cluster_3 = data[label == 2]

    centroid: Cluster centres

    """

    # observations
    data = np.array(get_list())

    # Find k clusters in the data
    centroid, label = kmeans2(data, k)

    for cluster in range(k):
        w = data[label == cluster]  # (array of array) boolean indexing of a list of list of stocks
        plt.figure()   #removes repeated figures

        for stock in w:  # list of data in one stock
            plt.plot(stock)

        plt.plot(centroid[cluster], c='r', label="centroid")
        plt.savefig("/Users/mcarmentz/Desktop/stock_screener/figures/" + str(cluster) + ".png")


def test_run():
    """Function called by Test Run."""
    k_means(3)  # k = 3 (see elbow_method.py)


if __name__ == "__main__":
    test_run()

