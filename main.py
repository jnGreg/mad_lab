
import numpy as np
import pandas as pd
from plotnine import *

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification

""" Wygenerowane wykresy: https://github.com/jnGreg/mad_lab/tree/1"""


def process_image(image_path) -> []:

    image = plt.imread(image_path, format='jpg')
    plt.axis('off')
    plt.imshow(image)
    # plt.show()

    # print(image.shape)  # (427, 640, 3)
    image_flat = image.reshape(-1, 3)
    # print(image_flat)
    return image_flat


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


def silhouette_index(flat_image):  # Indeks silhouette

    ks = range(2, 12)
    silhouettes = []

    for k in ks:
        km = KMeans(n_clusters=k).fit(flat_image)
        silhouettes.append(silhouette_score(flat_image, km.predict(flat_image)))

    silhouettes_df = pd.DataFrame({'K': ks,
                                   'Silhouette': silhouettes})

    g = (ggplot(silhouettes_df, aes(x='K', y='Silhouette')) +
         geom_point() + geom_line() +
         scale_x_continuous(breaks=ks) +
         theme_minimal() +
         labs(title="Silhouettes for subsequent K"))
    print(g)
    """ Zauważalny spadek wartości indeksu silhouette 
        ma miejsce dla K = 8, tym samym optymalna liczba
        klastrów wynosić powinna nie więcej, niż 7. 
        Jednakże w przypadku niedostatniego odwzorowania
        zdjęcia, warto spróbować także K = 9 lub K = 10,
        dla których indeks silhouette jest wyższy, niż dla K=8"""


def optimal_clusters_number(image_shape, flat_image, n):

    km = KMeans(n_clusters=n, random_state=0).fit(flat_image)
    flat_image_n = flat_image.copy()

    for i in np.unique(km.labels_):
        flat_image_n[km.labels_ == i, :] = km.cluster_centers_[i]
    image_n = flat_image_n.reshape(image_shape)
    plt.imshow(image_n)
    plt.show()


def main():
    image_path = 'pencils.jpg'
    flat_image = process_image(image_path)
    # scree_plot(flat_image)
    # silhouette_index(flat_image)
    image = plt.imread(image_path, format='jpg')

    #for n in range(3, 8):
        #optimal_clusters_number(image.shape, flat_image, n)

    """ Ponieważ celem zadania jest odwzorowanie obrazka
        przedstawiającego kolorowe kredki,
        minimalną liczbę klastrów ustalam na min. 3 
        (kolory podstawowe), a na podstawie wykresów osypiska
        oraz indeksu sylwetek oczekuję maksymalnej liczby 7."""

    optimal_clusters_number(image.shape, flat_image, 7)

    """  Po wygenerowaniu obrazów z liczbą klastrów od 3 do 7
      zauważam, że choć ok. 75% kredek jest właściwego koloru,
      warto spróbować jeszcze wyższej liczby klastrów. """

    #for n in range(7, 11):
        #optimal_clusters_number(image.shape, flat_image, n)

    """ Dla K = 7 oraz K = 8 różnica jest praktycznie niezauważalna, dla wyższych K
    następuje jednak zauważalne polepszenie odwzorowania zdjęcia, a za optymalną liczbę
    klastrów uznaję 10, ponieważ dopiero od tej liczby ostatnia para kredek 
    (sąsiadujące w lewym górnym rogu) jest rozróżnialna oraz posiadają one adekwatne barwy.
    Celem dalszego ograniczenia liczby klastrów można przyjąć także K = 7."""

    optimal_clusters_number(image.shape, flat_image, 10)


if __name__ == "__main__":
    main()
