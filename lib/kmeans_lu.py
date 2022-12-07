from typing import List

import numpy as np
import scipy.sparse as sp
from numba import njit
from sklearn.cluster._kmeans import _kmeans_plusplus
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _is_arraylike_not_scalar, check_array, check_random_state


@njit
def fast_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum((p1 - p2) ** 2))


def weighted_dist_python(p1: np.ndarray, p2: np.ndarray) -> float:
    diff_vect = (p1 - p2) ** 2
    diff_vect[:-1] *= 0.25
    diff_vect[-1] *= 0.75
    return np.sqrt(np.sum(diff_vect))


@njit
def weighted_dist_numba(p1: List[float], p2: List[float]) -> float:
    res_dist = 0
    for i in range(len(p1) - 1):
        res_dist += 0.25 * (p1[i] - p2[i]) ** 2
    res_dist += 0.75 * (p1[-1] - p2[-1]) ** 2
    return res_dist


def iter_python(
        X: np.ndarray,
        labels: List[int],
        centroids: np.ndarray,
        dist_func
):
    for i, pt in enumerate(X):
        labels[i] = int(np.argmin([dist_func(pt, centroid) for centroid in centroids]))

    for centroid_i, clust_label in enumerate(np.unique(labels)):
        cluster_points = X[np.where(labels == clust_label)]
        centroids[centroid_i] = cluster_points.mean(axis=0)


@njit
def iter_numba(
        X: np.ndarray,
        labels: List[int],
        centroids: np.ndarray,
        dist_func
):
    for i, pt in enumerate(X):
        min_dist = -1
        min_label = 0
        for centroid_label, centroid in enumerate(centroids):
            curr_dist = dist_func(pt, centroid)
            if min_dist == -1:
                min_dist = curr_dist
            elif curr_dist < min_dist:
                min_dist = curr_dist
                min_label = centroid_label

        labels[i] = min_label

    new_centroids = np.zeros(shape=(len(centroids), X.shape[1]))
    for clust_label in range(len(centroids)):
        centroid_i = clust_label
        clust_size = 0
        for pt, pt_label in zip(X, labels):
            if pt_label == clust_label:
                clust_size += 1
                for xi in range(X.shape[1]):
                    new_centroids[centroid_i][xi] += pt[xi]

        for xi in range(X.shape[1]):
            new_centroids[centroid_i][xi] /= clust_size

    for clust_label in range(len(centroids)):
        centroids[clust_label] = new_centroids[clust_label]


class KMeansLU:
    def __init__(self, n_clusters: int, random_state=None, max_iter=300):
        self.n_clusters = n_clusters
        # self._dist_func = dist

        # self.dist_func = weighted_dist_python
        # self.iter_func = iter_python
        self.dist_func = weighted_dist_numba
        self.iter_func = iter_numba

        self.random_state = check_random_state(random_state)
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None

    def _init_centroids(
            self, X, x_squared_norms, init, random_state, init_size=None, n_centroids=None
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters)

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X: np.ndarray):
        self.labels_ = np.zeros(shape=len(X))
        self.cluster_centers_ = self._init_centroids(
            X=X,
            x_squared_norms=row_norms(X, squared=True),
            init="k-means++",
            random_state=self.random_state
        )
        iter_i = 0
        for iter_i in range(self.max_iter):
            # print(f'iter = {iter_i}')
            self.iter_func(
                X=X,
                labels=self.labels_,
                centroids=self.cluster_centers_,
                dist_func=self.dist_func
            )

        if iter_i == self.max_iter:
            print("max_iter reached")

        return self

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).labels_

# for testing
# sorted([len(filt_df[filt_df['cl'] == cl]) for cl in filt_df['cl'].unique()])
