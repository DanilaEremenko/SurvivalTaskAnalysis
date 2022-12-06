from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from lib.kmeans_lu import fast_dist


class ClusteringBasedModel:
    def __init__(self, clust_key: str, cluster_centroids: pd.DataFrame):
        self.clust_key = clust_key
        self.models_dict: Optional[List[Dict]] = None
        self.cluster_centroids = cluster_centroids

    def fit_one_model(self, x_cl: pd.DataFrame, y_cl: np.ndarray) -> dict:
        common_args = {
            'n_estimators': [len(x_cl) // 500, len(x_cl) // 250],
            'bootstrap': [True, False],
            'random_state': [42]
        }
        param_grid = [
            {'max_depth': [2, 4, 8], **common_args},
            {'min_samples_leaf': [2, 4, 8], **common_args},
        ]
        clf_grid = GridSearchCV(RandomForestRegressor(), param_grid)
        clf_grid.fit(X=x_cl, y=y_cl)

        return {
            'params': clf_grid.best_params_,
            'score': clf_grid.best_score_,
            'model': clf_grid.best_estimator_
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        print('fitting cluster models')
        f_keys = [key for key in X.keys() if 'cl_l' not in key]
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
                    'dist': fast_dist(p1=centroid[[*f_keys, 'ElapsedRaw']].to_numpy(), p2=pt_coord)
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

        id_to_cluster_df = pd.DataFrame([
            {
                'id': i,
                'cl': self.get_closest_cluster(pt_coord=np.array([*x_features, curr_avg_y]), f_keys=f_keys)
            }
            for (i, x_features), curr_avg_y in zip(X[f_keys].iterrows(), y_avg)]
        ).set_index('id')
        id_to_cluster_df['y_pred'] = -1
        for model_cluster, model_dict in self.models_dict.items():
            model = model_dict['model']
            model_indexes = id_to_cluster_df[id_to_cluster_df['cl'] == model_cluster].index
            model_X = X.iloc[model_indexes]
            id_to_cluster_df.loc[model_indexes, 'y_pred'] = model.predict(model_X)

        y_selected = np.array(id_to_cluster_df['y_pred'])
        assert -1 not in y_selected

        return y_selected
