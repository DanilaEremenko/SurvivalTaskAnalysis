import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class ClusteringBasedModel:
    def __init__(self, clust_key: str):
        self.clust_key = clust_key

    def fit_one_model(self, x_cl: pd.DataFrame, y_cl: np.ndarray):
        common_args = {'random_state': [42]}
        param_grid = [
            {'n_estimators': [100, 250, 500], 'max_depth': [2, 4, 8], **common_args},
            {'n_estimators': [100, 250, 500], 'min_samples_leaf': [2, 4, 8], **common_args},
        ]
        clf_grid = GridSearchCV(RandomForestRegressor(), param_grid)
        clf_grid.fit(X=x_cl, y=y_cl)

        return clf_grid.best_estimator_

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        models_list = [
            self.fit_one_model(X, y)
            for clust_val in X[self.clust_key].unique()
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return []
