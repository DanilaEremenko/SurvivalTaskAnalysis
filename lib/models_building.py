import json
import random
import re
import time
from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid, cross_val_score, KFold
from sksurv.ensemble import RandomSurvivalForest

from experiments_config import EXP_PATH, CL_MODE, CL_DIR
from lib.custom_models import ClusteringBasedModel
from lib.custom_survival_funcs import get_event_time_manual, batch_surv_time_pred, batch_risk_score_pred, \
    get_t_from_y
from lib.losses import Losses


def get_reg_predictions_and_metrics_df(model, X, y) -> pd.DataFrame:
    res_test_df = pd.DataFrame(
        {
            'y_true': y,
            'y_pred': model.predict(X=X),
        }
    )

    res_test_df = pd.merge(left=X, right=res_test_df, left_index=True, right_index=True)

    res_test_df['residual'] = res_test_df['y_true'] - res_test_df['y_pred']
    res_test_df['residual perc'] = res_test_df['residual'] / res_test_df['y_true'] * 100.

    res_test_df['mae'] = res_test_df['residual'].abs()
    res_test_df['mae perc'] = res_test_df['residual perc'].abs()
    return res_test_df


ASSYM_COEF = 10.


def assym_obj_fn(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual > 0, -2.0 * ASSYM_COEF * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * ASSYM_COEF, 2.0)
    return grad, hess


def assym_valid_fn(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual > 0, (residual ** 2.0) * ASSYM_COEF, residual ** 2.0)
    return "custom_asymmetric_eval", np.mean(loss), False


def build_scenarios(
        x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
        method: str
) -> pd.DataFrame:
    res_list = []

    if method == 'rf':
        common_args = {'n_estimators': [len(x_train) // ex_in_trees for ex_in_trees in (500, 1000)],
                       'bootstrap': [True], 'max_features': [0.5, 1.0],
                       'random_state': [42]}
        params_grid = [
            # {'max_depth': [2, 4, 8], **common_args},
            # {'min_samples_leaf': [1, 2, 4, 8], **common_args},
            {'min_samples_leaf': [1, 2, 4, 8], 'max_depth': [10, 20, 30], **common_args},

        ]
        params_list = list(ParameterGrid(params_grid))
        for i, args_dict in enumerate(params_list):
            print(f"fitting scenario {i}/{len(params_list)}")

            model = RandomForestRegressor(**args_dict)
            model.fit(X=x_train, y=y_train)

            print('testing')
            # res_test_df = get_reg_predictions_and_metrics_df(model=model, X=x_test, y=y_test)
            # mae = res_test_df['mae perc'].mean()
            cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    # 'mae': mae,
                    'r2_mean_train_cv': cv_score.mean(),
                    'r2_std_train_cv': cv_score.std(),
                    'r2_mean_std': cv_score.mean() / cv_score.std(),
                    'r_test': Losses.r(pred=model.predict(x_test), y=y_test)
                }
            )

    elif method == 'lgbm':
        common_args = {'n_estimators': [len(x_train) // ex_in_trees for ex_in_trees in (500, 1000)],
                       'random_state': [42]}
        params_grid = [
            # {'max_depth': [2, 4, 8, 16, 32], **common_args},
            # {'min_child_samples': [1, 2, 4, 8], **common_args}
            {'min_child_samples': [1, 2, 4, 8], 'max_depth': [10, 20, 30, 40, 50], **common_args},
        ]
        params_list = list(ParameterGrid(params_grid))
        for i, args_dict in enumerate(params_list):
            print(f"fitting scenario {i}/{len(params_list)}")
            model = LGBMRegressor(**args_dict)
            model.set_params(objective=assym_obj_fn)
            model.fit(X=x_train.to_numpy(), y=y_train, eval_metric=assym_valid_fn)

            x_train = x_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

            print('testing')
            # res_test_df = get_reg_predictions_and_metrics_df(model=model, X=x_test, y=y_test)
            # mae = res_test_df['mae perc'].mean()
            cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'r2_mean_train_cv': cv_score.mean(),
                    'r2_std_train_cv': cv_score.std(),
                    'r2_mean_std': cv_score.mean() / cv_score.std(),
                    'r_test': Losses.r(pred=model.predict(x_test), y=y_test)
                }
            )

    elif method == 'rsf':
        # rsf predicts are working too long
        random.seed(42)
        test_subset_indexes = np.random.randint(low=0, high=len(x_test), size=10_000)
        x_test = x_test.iloc[test_subset_indexes]
        y_test = y_test[test_subset_indexes]

        common_args = {'n_estimators': [len(x_train) // ex_in_trees for ex_in_trees in (500, 1000)],
                       'bootstrap': [True], 'max_features': [1.0],
                       'max_samples': [500], 'random_state': [42]}
        params_grid = [
            # {'max_depth': [2, 4, 8], **common_args},
            # {'min_samples_leaf': [2, 4, 8], **common_args},
            {'min_samples_leaf': [2, 4, 8], 'max_depth': [10], **common_args},
        ]
        params_list = list(ParameterGrid(params_grid))
        for i, args_dict in enumerate(params_list):
            print(f"fitting scenario {i + 1}/{len(params_list)}")

            model = RandomSurvivalForest(**args_dict)
            start_time = time.time()
            model.fit(X=x_train, y=y_train)
            fit_time = time.time() - start_time

            print('testing')
            start_time = time.time()
            y_test_pred_time = batch_surv_time_pred(model=model, X=x_test)
            time_calc = time.time() - start_time

            kf = KFold(n_splits=5, shuffle=False)
            cv_score = []
            for i, (cv_train_index, cv_test_index) in enumerate(kf.split(x_train)):
                print(f'cross validation {i + 1}/5')
                model.fit(x_train.iloc[cv_train_index], y_train[cv_train_index])
                cv_score.append(
                    r2_score(
                        y_true=get_t_from_y(y_train[cv_test_index]),
                        y_pred=batch_surv_time_pred(model=model, X=x_train.iloc[cv_test_index])
                    )
                )
            cv_score = np.array(cv_score)

            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'r2_mean_train_cv': cv_score.mean(),
                    'r2_std_train_cv': cv_score.std(),
                    'r2_mean_std': cv_score.mean() / cv_score.std(),
                    'r_test': Losses.r(pred=y_test_pred_time, y=get_t_from_y(y_test))
                }
            )

    elif method == 'cb':
        for cl_lvl in [1, 2, 3]:
            print(f'fitting on cl_lvl = {cl_lvl}')
            model = ClusteringBasedModel(
                clust_key=f'cl_l{cl_lvl}',
                cluster_centroids=pd.read_csv(
                    f'{EXP_PATH}/clustering_{CL_MODE}_{CL_DIR}/train_centroids_l{cl_lvl}.csv',
                    index_col=0
                )
            )
            print('predicting')
            start_time = time.time()
            model.fit(X=x_train, y=y_train)
            y_test_pred = model.predict(X=x_test)

            kf = KFold(n_splits=5, shuffle=False)
            cv_score = []
            for i, (cv_train_index, cv_test_index) in enumerate(kf.split(x_train)):
                print(f'cross validation {i + 1}/5')
                model.fit(x_train.iloc[cv_train_index], y_train[cv_train_index])
                cv_score.append(
                    r2_score(
                        y_true=get_t_from_y(y_train[cv_test_index]),
                        y_pred=model.predict(x_train.iloc[cv_test_index])
                    )
                )
            cv_score = np.array(cv_score)

            res_list.append(
                {
                    'args_dict': json.dumps({'cl_lvl': cl_lvl}),
                    'fit_predict_time': time.time() - start_time,
                    'r2_mean_train_cv': cv_score.mean(),
                    'r2_std_train_cv': cv_score.std(),
                    'r2_mean_std': cv_score.mean() / cv_score.std(),
                    'r_test': Losses.r(pred=y_test_pred, y=get_t_from_y(y_test))
                }
            )

    else:
        raise Exception(f'Undefined reg methods = {method}')

    return pd.DataFrame(res_list)
