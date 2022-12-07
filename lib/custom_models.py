from collections import Counter
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from experiments import CL_MODE
from lib.kmeans_lu import fast_dist, weighted_dist_numba
from lib.time_ranges import get_time_range_symb


class ClusteringBasedModel:
    def __init__(self, clust_key: str, cluster_centroids: pd.DataFrame):
        self.clust_key = clust_key
        self.models_dict: Optional[List[Dict]] = None
        self.cluster_centroids = cluster_centroids
        self.norm_dict: Optional[Dict[str, MinMaxScaler]] = None

        if CL_MODE == 'km':
            self._dist_func = fast_dist
        elif CL_MODE == 'kmlu':
            self._dist_func = weighted_dist_numba
        else:
            raise Exception(f"Unexpected clustering mode = {CL_MODE}")

    def fit_one_model(self, x_cl: pd.DataFrame, y_cl: np.ndarray) -> dict:
        common_args = {
            'n_estimators': [max(len(x_cl) // n_samples, 5) for n_samples in [500, 1000]],
            'bootstrap': [True],
            'random_state': [42]
        }
        param_grid = [
            {'max_depth': [2, 4, 8], **common_args},
            {'min_samples_leaf': [1, 2, 4], **common_args},
        ]
        clf_grid = GridSearchCV(RandomForestRegressor(), param_grid)

        print('fitting model')
        print(f'cluster_size={len(x_cl)}')

        cl_dist = dict(Counter([get_time_range_symb(task_time=task_time) for task_time in y_cl]))
        cl_dist_df = pd.DataFrame({'time_range': cl_dist.keys(), 'tasks(%)': cl_dist.values()}) \
            .sort_values('time_range', ascending=True)
        cl_dist_df['tasks(%)'] /= len(y_cl)
        print(f'classes distribution in cluster: \n{cl_dist_df.round(2)}')

        clf_grid.fit(X=x_cl, y=y_cl)
        print(f'best grid params = {clf_grid.best_params_}')
        print(f'best grid score = {round(clf_grid.best_score_, 2)}')

        return {
            # 'params': clf_grid.best_params_,
            # 'score': clf_grid.best_score_,
            'model': clf_grid.best_estimator_
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        f_keys = [key for key in X.keys() if 'cl_l' not in key]

        # fit normalizers on train data
        self.norm_dict: Dict[str, MinMaxScaler] = {
            key: MinMaxScaler().fit(np.array(X[key]).reshape(-1, 1))
            for key in f_keys
        }
        self.norm_dict['ElapsedRaw'] = MinMaxScaler().fit(np.array(y).reshape(-1, 1))

        # normalize cluster centroids
        for key, norm in self.norm_dict.items():
            self.cluster_centroids[key] = norm.transform(np.array(self.cluster_centroids[key]).reshape(-1, 1))

        self.models_dict = {
            clust_val: self.fit_one_model(
                X[X[self.clust_key] == clust_val][f_keys],
                y[X[X[self.clust_key] == clust_val].index]
            )
            for clust_val in X[self.clust_key].unique()
        }
        return self

    def get_closest_cluster(self, pt_coord: np.ndarray, f_keys: List[str]) -> int:
        clust_distances_df = pd.DataFrame(
            [
                {
                    'cl': centroid['cl'],
                    'dist': weighted_dist_numba(p1=centroid[[*f_keys, 'ElapsedRaw']].to_numpy(), p2=pt_coord)
                }
                for i, centroid in self.cluster_centroids.iterrows()
            ]
        )
        return clust_distances_df.sort_values('dist', ascending=True).iloc[0]['cl']

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.models_dict is None:
            raise Exception("Trying to call predict without fitting")

        f_keys = [key for key in X.keys() if 'cl_l' not in key]
        y_pred_all = np.array([model_dict['model'].predict(X[f_keys]) for model_dict in self.models_dict.values()])
        y_avg = y_pred_all.mean(axis=0)

        X_norm_and_extended = X.copy()
        X_norm_and_extended['ElapsedRaw'] = y_avg
        for key, norm in self.norm_dict.items():
            X_norm_and_extended[key] = norm.transform(np.array(X_norm_and_extended[key]).reshape(-1, 1))

        print('calulating id to cluster')
        id_to_cluster_df = pd.DataFrame([
            {
                'id': i,
                'cl': self.get_closest_cluster(pt_coord=pt_features.to_numpy(), f_keys=f_keys)
            }
            for i, pt_features in X_norm_and_extended[[*f_keys, 'ElapsedRaw']].iterrows()]
        ).set_index('id')
        id_to_cluster_df['y_pred'] = -1
        print('each model predicting')
        for model_cluster, model_dict in self.models_dict.items():
            model = model_dict['model']
            model_indexes = id_to_cluster_df[id_to_cluster_df['cl'] == model_cluster].index
            if len(model_indexes) == 0:
                continue

            model_X = X.iloc[model_indexes]
            id_to_cluster_df.loc[model_indexes, 'y_pred'] = model.predict(model_X)

        y_selected = np.array(id_to_cluster_df['y_pred'])
        assert -1 not in y_selected

        return y_selected
