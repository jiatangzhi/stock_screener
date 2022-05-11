import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from stock_data import get_list

def elbow_method():
    """
       Return a graph to identify optimal value of k.
    """

    sum_of_squared_distances = []  # Error Sum of Squares(sse).
    k = range(1, 10)
    for num_clusters in k:
        km = KMeans(n_clusters=num_clusters)
        km.fit(get_list())
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(k, sum_of_squared_distances)
    plt.xlabel("Values of K")
    plt.ylabel("Sum of squared distances/Inertia")
    plt.title("Elbow Method For Optimal k")
    plt.savefig("/Users/mcarmentz/Desktop/stock_screener/figures/elbow_method.png")


if __name__ == "__main__":
    elbow_method()     # optimal value of k = 3 (see elbow_method.png)
