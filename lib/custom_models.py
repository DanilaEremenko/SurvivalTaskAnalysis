from collections import Counter
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest

from experiments_config import CL_MODE, CL_CENTROIDS_DIST_MODE
from lib.custom_reg import assym_obj_fn, assym_valid_fn
from lib.custom_survival_funcs import batch_surv_time_pred, get_t_from_y
from lib.kmeans_lu import fast_dist, weighted_dist_numba
from lib.time_ranges import get_time_range_symb
from prepare_clust import print_one_cl_dist


class ClusteringBasedModel:
    def __init__(self, clust_key: str, cluster_centroids: pd.DataFrame):
        self.clust_key = clust_key
        self.models_dict: Dict[str, Dict] = None
        self._cluster_centroids = cluster_centroids
        self._cluster_centroids_surv = None
        self.norm_dict: Optional[Dict[str, MinMaxScaler]] = None
        self.normalize = False
        self.regressor = LGBMRegressor()

        if CL_MODE == 'km':
            self._dist_func = fast_dist
        elif CL_MODE == 'kmlu':
            self._dist_func = weighted_dist_numba
        else:
            raise Exception(f"Unexpected clustering mode = {CL_MODE}")

    @property
    def cluster_centroids_dist(self) -> pd.DataFrame:
        if CL_CENTROIDS_DIST_MODE == 'orig(y)':
            return self._cluster_centroids
        elif CL_CENTROIDS_DIST_MODE == 'y_pred(y)':
            return self._cluster_centroids_surv
        else:
            raise Exception(f"Undefined CL_CENTROIDS_DIST_MODE = {CL_CENTROIDS_DIST_MODE}")

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

        # fit normalizers on train data_type
        self.norm_dict: Dict[str, MinMaxScaler] = {
            key: MinMaxScaler().fit(np.array(X[key]).reshape(-1, 1))
            for key in f_keys
        }
        # self.norm_dict['ElapsedRaw'] = MinMaxScaler().fit(np.array(y).reshape(-1, 1))

        # normalize cluster centroids
        for key, norm in self.norm_dict.items():
            self._cluster_centroids[key] = self.normalize_series(self._cluster_centroids[key], norm)

        X_idx_r = X.reset_index(drop=True)
        self.models_dict = {
            clust_val: self.fit_one_model(
                X_idx_r[X_idx_r[self.clust_key] == clust_val][f_keys],
                y[X_idx_r[X_idx_r[self.clust_key] == clust_val].index],
                full_train_size=len(X)
            )
            for clust_val in self._cluster_centroids['cl']
        }

        self.regressor = LGBMRegressor(max_depth=10, min_child_samples=2, n_estimators=27, random_state=42)
        self.regressor.set_params(objective=assym_obj_fn)
        self.regressor.fit(X=X[f_keys].to_numpy(), y=get_t_from_y(y), eval_metric=assym_valid_fn)

        self._cluster_centroids_surv = self._cluster_centroids.copy()
        for i, (id, cl) in enumerate(self._cluster_centroids_surv[['cl']].iterrows()):
            cl = int(cl)
            self._cluster_centroids_surv.loc[id, 'ElapsedRaw'] = batch_surv_time_pred(
                self.models_dict[cl]['model'],
                self._cluster_centroids[f_keys],
                mode='math_exp'
            )[i]

        return self

    def get_closest_cluster(self, pt_coord: np.ndarray, f_keys: List[str]) -> int:
        clust_distances_df = pd.DataFrame(
            [
                {
                    'cl': centroid['cl'],
                    'dist': weighted_dist_numba(p1=centroid[[*f_keys, 'ElapsedRaw']].to_numpy(), p2=pt_coord)
                }
                # for i, centroid in self._cluster_centroids_surv.iterrows()
                for i, centroid in self.cluster_centroids_dist.iterrows()
            ]
        )
        return clust_distances_df.sort_values('dist', ascending=True).iloc[0]['cl']

    def predict(self, X: pd.DataFrame, y_test) -> np.ndarray:
        if self.models_dict is None:
            raise Exception("Trying to call predict without fitting")

        f_keys = [key for key in X.keys() if 'cl_l' not in key]
        y_pred_all = np.array([batch_surv_time_pred(model_dict['model'], X[f_keys], mode='math_exp')
                               for model_dict in self.models_dict.values()])
        y_avg = y_pred_all.mean(axis=0)
        y_reg = self.regressor.predict(X[f_keys])
        # return y_avg

        X_norm_and_extended = X.copy()
        X_norm_and_extended['ElapsedMeanSurv'] = y_avg
        X_norm_and_extended['ElapsedReg'] = y_reg

        for key, norm in self.norm_dict.items():
            X_norm_and_extended[key] = self.normalize_series(X_norm_and_extended[key], norm)

        print('calulating id to cluster')

        id_to_cluster_avg_df = pd.DataFrame([
            {
                'pt_id': pt_id,
                'cl': self.get_closest_cluster(pt_coord=pt_features.to_numpy(), f_keys=f_keys)
            }
            for pt_id, pt_features in X_norm_and_extended[[*f_keys, 'ElapsedMeanSurv']].iterrows()]
        ).set_index('pt_id')

        id_to_cluster_reg_df = pd.DataFrame([
            {
                'pt_id': pt_id,
                'cl': self.get_closest_cluster(pt_coord=pt_features.to_numpy(), f_keys=f_keys)
            }
            for pt_id, pt_features in X_norm_and_extended[[*f_keys, 'ElapsedReg']].iterrows()]
        ).set_index('pt_id')

        id_to_cluster_avg_df['y_pred'] = -1
        id_to_cluster_reg_df['y_pred'] = -1

        print('each model predicting')
        for model_cluster, model_dict in self.models_dict.items():
            model = model_dict['model']

            # choose model using different cluster defining approaches
            for id_to_cluster_var in [id_to_cluster_avg_df, id_to_cluster_reg_df]:
                model_indexes = id_to_cluster_var[id_to_cluster_var['cl'] == model_cluster].index

                if len(model_indexes) == 0:
                    continue

                id_to_cluster_var.loc[model_indexes, 'y_pred'] = batch_surv_time_pred(
                    model,
                    X.loc[model_indexes, f_keys],
                    mode='math_exp'
                )

        for id_to_cluster_var in [id_to_cluster_avg_df, id_to_cluster_reg_df]:
            assert -1 not in np.array(id_to_cluster_var['y_pred'])

        # self._debug_arr = np.column_stack((*[y_pred_model for i, y_pred_model in enumerate(y_pred_all)],
        #                                    y_avg, y_selected, get_t_from_y(y_test),
        #                                    id_to_cluster_df['cl']))
        self._debug_df = pd.DataFrame(
            {
                **{f'y_cl{i}': y_pred_model for i, y_pred_model in enumerate(y_pred_all)},
                'y_avg': y_avg,
                'y_selected_avg': np.array(id_to_cluster_avg_df['y_pred']),
                'y_reg': y_reg,
                'y_selected_reg': np.array(id_to_cluster_reg_df['y_pred']),
                'y_true': get_t_from_y(y_test),
                **{f'y_cl{i}_resid': y_pred_model - get_t_from_y(y_test)
                   for i, y_pred_model in enumerate(y_pred_all)},
                'cl_avg': id_to_cluster_avg_df['cl'],
                'cl_avg_y': list(
                    pd.merge(left=id_to_cluster_avg_df.reset_index(),
                             right=self._cluster_centroids[['cl', 'ElapsedRaw']],
                             on='cl'
                             ).set_index('pt_id').loc[id_to_cluster_avg_df.index, 'ElapsedRaw']),
                'cl_reg': id_to_cluster_avg_df['cl'],
                'cl_reg_y': list(
                    pd.merge(left=id_to_cluster_reg_df.reset_index(),
                             right=self._cluster_centroids[['cl', 'ElapsedRaw']],
                             on='cl'
                             ).set_index('pt_id').loc[id_to_cluster_avg_df.index, 'ElapsedRaw'])
            }
        )

        return np.array(id_to_cluster_reg_df['y_pred'])
