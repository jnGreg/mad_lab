
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification


def lab():
    # Code written during the lab
    n = 1000

    centra = [(-6, -6), (0, 0), (9, 9)]

    # blob
    X, y = make_blobs(n_samples=n, n_features=2, centers=centra, cluster_std=1, random_state=0)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    km = KMeans(n_clusters=4, random_state=0)
    km.fit(X)

    km.labels_

    plt.scatter(X[:, 0], X[:, 1], c=km.labels_)

    km.cluster_centers_

    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="white")

    # classification
    X, y = make_classification(n_samples=n, n_features=2, n_clusters_per_class=1, n_redundant=0, random_state=4)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    km = KMeans(n_clusters=2, random_state=0).fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=km.labels_)

    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="red")
    plt.show()

    most = plt.imread('most.jpg', format='jpg')
    plt.imshow(most)
    plt.axis('off')

    # most.shape
    # most_flat = most.reshape(427*640, 3)
    # most_flat = most.reshape(most.shape[0]*most.shape[1], 3)
    most_flat = most.reshape(-1, 3)

    km = KMeans(n_clusters=4, random_state=0).fit(most_flat)

    km.labels_
    np.unique(km.labels_)

    km.cluster_centers_

    most_flat4 = most_flat.copy()
    for i in np.unique(km.labels_):
        most_flat4[km.labels_ == i, :] = km.cluster_centers_[i]

    most4 = most_flat4.reshape(most.shape)
    plt.imshow(most4)
    plt.axis('off')


def prepare_sample_data() -> [[]]:
    # Simple sample array of [10 to 20] by [10 to 20] objects with a value from 0 to 1.
    sample_data = np.random.rand(np.random.randint(10, 21), np.random.randint(10, 21))
    return sample_data


def prepare_sample_data() -> [[]]:
    # Simple sample array of [10 to 20] by [10 to 20] objects with a value from 0 to 1.
    sample_data = np.random.rand(np.random.randint(10, 21), np.random.randint(10, 21))
    return sample_data


def min_max(sample_data) -> [[]]:
    # Takes given array, returns array normalized with min max method
    # min max method: (value - min) / (max - min)
    mm_norm = []
    for row in sample_data:
        mm_row = []
        [mm_row.append((value-min(row))/(max(row)-min(row))) for value in row]
        mm_norm.append(mm_row)
    mm_norm = np.array(mm_norm)
    return mm_norm


def standardization(sample_data) -> [[]]:
    # Takes given array, returns array normalized with standardization method
    # standardization method: value - mean / std
    std_norm = []
    for row in sample_data:
        std_row = []
        [std_row.append((value-np.mean(row))/np.std(row)) for value in row]
        std_norm.append(std_row)
    std_norm = np.array(std_norm)
    return std_norm


def main():
    sample_data = prepare_sample_data()
    print(sample_data)
    mm_norm = min_max(sample_data)
    print(mm_norm)
    std_norm = standardization(sample_data)
    print(std_norm)


if __name__ == "__main__":
    main()
