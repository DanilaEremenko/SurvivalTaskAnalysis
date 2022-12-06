from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class ClusteringBasedModel:
    def __init__(self, clust_key: str):
        self.clust_key = clust_key
        self.models_list: Optional[List[Dict]] = None

    def fit_one_model(self, x_cl: pd.DataFrame, y_cl: np.ndarray) -> dict:
        common_args = {
            'n_estimators': [10],
            'bootstrap': [True],
            # 'n_estimators': [len(x_cl) // 500, len(x_cl) // 250],
            # 'bootstrap': [True, False],
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
        features = [key for key in X.keys() if 'cl_l' not in key]
        self.models_list = [
            self.fit_one_model(X[X[self.clust_key] == clust_val][features], y[X[X[self.clust_key] == clust_val].index])
            for clust_val in X[self.clust_key].unique()
        ]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = [key for key in X.keys() if 'cl_l' not in key]
        avg_y = np.array([model_dict['model'].predict(X[features]) for model_dict in self.models_list]).mean(axis=0)
        return avg_y
