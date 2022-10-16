
import numpy as np
import pandas as pd
from plotnine import *

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification


def read_image(image) -> []:

    pencils = plt.imread(image, format='jpg')
    plt.axis('off')
    plt.imshow(pencils)
    # plt.show()

    # print(pencils.shape)  # (427, 640, 3)
    pencils_flat = pencils.reshape(-1, 3)
    # print(pencils_flat)
    return pencils_flat


def scree_plot(flat_image):  # Wykres osypiska

    ks = range(2, 20)
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k).fit(flat_image)
        inertias.append(km.inertia_)

    inertias_df = pd.DataFrame({'K': ks,
                                'Inertia': inertias})

    g = (ggplot(inertias_df, aes(x='K', y='Inertia')) +
         geom_point() + geom_line() +
         scale_x_continuous(breaks=ks) +
         theme_minimal() +
         labs(title="Inertias for subsequent K"))
    print(g)
    """ Spadek przyrostu inercji następuje w 5 K,
        jednak przegięcie to nie jest wystarczająco silne,
        by na podstawie wykresu osypiska stwierdzić liczbę
        klastrów (ich liczba wynosić może od 5 do 10) """


def main():
    scree_plot(read_image('pencils.jpg'))


if __name__ == "__main__":
    main()
