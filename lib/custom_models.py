from collections import Counter
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest

from experiments import CL_MODE
from lib.custom_survival_funcs import batch_surv_time_pred, get_t_from_y
from lib.kmeans_lu import fast_dist, weighted_dist_numba
from lib.time_ranges import get_time_range_symb
from prepare_clust import print_one_cl_dist


class ClusteringBasedModel:
    def __init__(self, clust_key: str, cluster_centroids: pd.DataFrame):
        self.clust_key = clust_key
        self.models_dict: Optional[List[Dict]] = None
        self.cluster_centroids = cluster_centroids
        self.norm_dict: Optional[Dict[str, MinMaxScaler]] = None
        self.normalize = False

        if CL_MODE == 'km':
            self._dist_func = fast_dist
        elif CL_MODE == 'kmlu':
            self._dist_func = weighted_dist_numba
        else:
            raise Exception(f"Unexpected clustering mode = {CL_MODE}")

    def fit_one_model(self, x_cl: pd.DataFrame, y_cl: np.ndarray, full_train_size: int) -> dict:
        cv = 5

        common_args = {
            'n_estimators': [max(5, len(x_cl) // ex_in_trees) for ex_in_trees in (500, 1000)],
            # 'n_estimators': [10],
            'bootstrap': [True], 'max_features': [1.0],
            'max_samples': [min(500, int(len(x_cl) * (1 - 1 / cv)))], 'random_state': [42]}
        params_grid = [
            # {'max_depth': [2, 4, 8], **common_args},
            # {'min_samples_leaf': [2, 4, 8], **common_args},
            {'min_samples_leaf': [4], 'max_depth': [10], **common_args},
        ]
        clf_grid = GridSearchCV(RandomSurvivalForest(), params_grid, cv=cv)

        print('fitting model')
        print(f'cl size = {len(x_cl)}')
        print_one_cl_dist(pd.DataFrame({'ElapsedRaw': get_t_from_y(y_cl)}))

        clf_grid.fit(X=x_cl, y=y_cl)
        print(f'best grid params = {clf_grid.best_params_}')
        print(f'best grid score = {round(clf_grid.best_score_, 2)}')

        return {
            # 'params': clf_grid.best_params_,
            # 'score': clf_grid.best_score_,
            'model': clf_grid.best_estimator_
        }

    def normalize_series(self, s: pd.Series, norm: MinMaxScaler, ):
        if self.normalize:
            return norm.transform(np.array(s).reshape(-1, 1))
        else:
            return s

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        f_keys = [key for key in X.keys() if 'cl_l' not in key]

        # fit normalizers on train data
        self.norm_dict: Dict[str, MinMaxScaler] = {
            key: MinMaxScaler().fit(np.array(X[key]).reshape(-1, 1))
            for key in f_keys
        }
        # self.norm_dict['ElapsedRaw'] = MinMaxScaler().fit(np.array(y).reshape(-1, 1))

        # normalize cluster centroids
        for key, norm in self.norm_dict.items():
            self.cluster_centroids[key] = self.normalize_series(self.cluster_centroids[key], norm)

        X_idx_r = X.reset_index(drop=True)
        self.models_dict = {
            clust_val: self.fit_one_model(
                X_idx_r[X_idx_r[self.clust_key] == clust_val][f_keys],
                y[X_idx_r[X_idx_r[self.clust_key] == clust_val].index],
                full_train_size=len(X)
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
        y_pred_all = np.array([batch_surv_time_pred(model_dict['model'], X[f_keys])
                               for model_dict in self.models_dict.values()])
        y_avg = y_pred_all.mean(axis=0)

        # return y_avg

        X_norm_and_extended = X.copy()
        X_norm_and_extended['ElapsedRaw'] = y_avg
        for key, norm in self.norm_dict.items():
            X_norm_and_extended[key] = self.normalize_series(X_norm_and_extended[key], norm)

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

            model_X = X.loc[model_indexes]
            id_to_cluster_df.loc[model_indexes, 'y_pred'] = batch_surv_time_pred(model, model_X)

        y_selected = np.array(id_to_cluster_df['y_pred'])
        assert -1 not in y_selected

        # debug_arr = np.column_stack((y_pred_all[0], y_pred_all[1], y_avg,
        #                              y_selected, get_t_from_y(y_test),
        #                              id_to_cluster_df['cl']))
        # debug_df = pd.DataFrame({'y_1': y_pred_all[0], 'y_2': y_pred_all[1], 'y_avg': y_avg,
        #                          'y_selected': y_selected, 'y_true': get_t_from_y(y_test),
        #                          'cl': id_to_cluster_df['cl']})

        return y_selected
