import json
import random
import time
from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import RandomSurvivalForest

from experiments import EXP_PATH, CL_MODE
from lib.custom_models import ClusteringBasedModel
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


def get_event_time_manual(event_times: np.ndarray, probs: np.ndarray) -> List[float]:
    res_arr = []
    print('predicting 200')
    for i, curr_probs in enumerate(probs):
        for et, prob in zip(event_times, curr_probs):
            if prob == curr_probs.min():
                res_arr.append(et)
                break
        # if len(res_arr) != i + 1:
        #     raise Exception('No prob < 0.1 in prob vector')
    return res_arr


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
            res_test_df = get_reg_predictions_and_metrics_df(model=model, X=x_test, y=y_test)
            mae = res_test_df['mae perc'].mean()
            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'mae': mae,
                    'r': Losses.r(pred=res_test_df['y_pred'], y=res_test_df['y_true'])
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

            print('testing')
            res_test_df = get_reg_predictions_and_metrics_df(model=model, X=x_test, y=y_test)
            mae = res_test_df['mae perc'].mean()
            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'mae': mae,
                    'r': Losses.r(pred=res_test_df['y_pred'], y=res_test_df['y_true'])
                }
            )

    elif method == 'rsf':
        # rsf predicts are working too long
        random.seed(42)
        test_subset_indexes = np.random.randint(low=0, high=len(x_test), size=10_000)

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
            print(f"fitting scenario {i}/{len(params_list)}")

            model = RandomSurvivalForest(**args_dict)
            start_time = time.time()
            model.fit(X=x_train, y=y_train)

            print('testing')
            y_test_pred = model.predict(x_test.iloc[test_subset_indexes])

            # y_test_pred = np.concatenate([
            #     get_event_time_manual(
            #         event_times=model.event_times_,
            #         probs=model.predict_survival_function(x_test[start:start + 200], return_array=True)
            #     )
            #     for start in range(0, len(x_test), 200)
            # ])

            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'fit_predict_time': time.time() - start_time,
                    # 'c-val': model.score(X=x_test[:10_000], y=y_test[:10_000]),
                    # 'c-val': model.score(X=x_test, y=y_test),
                    'r': Losses.r(y=[y[1] for y in y_test[test_subset_indexes]], pred=y_test_pred)
                }
            )
    elif method == 'cb':
        for cl_lvl in [1, 2, 3, 4]:
            print(f'fitting on cl_lvl = {cl_lvl}')
            model = ClusteringBasedModel(
                clust_key=f'cl_l{cl_lvl}',
                cluster_centroids=pd.read_csv(
                    f'{EXP_PATH}/clustering_{CL_MODE}/train_centroids_l{cl_lvl}.csv',
                    index_col=0
                )
            )
            print('predicting')
            start_time = time.time()
            model.fit(X=x_train, y=y_train)
            y_test_pred = model.predict(X=x_test)
            res_list.append(
                {
                    'args_dict': json.dumps({'cl_lvl': cl_lvl}),
                    'fit_predict_time': time.time() - start_time,
                    'r': Losses.r(y=y_test, pred=y_test_pred)
                }
            )
    else:
        raise Exception(f'Undefined reg methods = {method}')

    return pd.DataFrame(res_list)
