import numpy as np
from sklearn.cluster import KMeans

from lib.kmeans_lu import KMeansLU

X = np.zeros(shape=(1000, 2))
X[0:300] += np.concatenate([np.random.normal(10, 5, (300, 1)), np.random.normal(10, 5, (300, 1))], axis=1)
X[300:1000] += np.concatenate([np.random.normal(30, 5, (700, 1)), np.random.normal(10, 5, (700, 1))], axis=1)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.show()

algs = [
    {
        'name': 'KMeans',
        'func': KMeans(n_clusters=2, random_state=42).fit_predict
    },
    {
        'name': 'KMeansLU',
        'func': KMeansLU(n_clusters=2, random_state=42).fit_predict
    }
]

for alg_dict in algs:
    labels = alg_dict['func'](X)
    for label in np.unique(labels):
        curr_pts = X[np.where(labels == label)]
        plt.scatter(curr_pts[:, 0], curr_pts[:, 1])
    plt.title(alg_dict['name'])
    plt.show()
