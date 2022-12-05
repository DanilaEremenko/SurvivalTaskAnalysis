import json
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sksurv.ensemble import RandomSurvivalForest

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


def build_scenarios(
        x_train, y_train, x_test, y_test,
        args_scenarios: List[dict],
        method: str
) -> pd.DataFrame:
    res_list = []

    for i, args_dict in enumerate(args_scenarios):
        print(f'fitting scenario {i}/{len(args_scenarios)}')

        if method == 'rf':
            model = RandomForestRegressor(**args_dict)
            model.fit(X=x_train, y=y_train)
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
            model = RandomSurvivalForest(**args_dict)
            start_time = time.time()
            model.fit(X=x_train, y=y_train)
            print('predicting')
            y_test_pred = model.predict(x_test)

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
                    # 'c-val': model.score(X=x_test[:10_000], y=y_test[:10_000]),
                    'r': Losses.r(y=[y[1] for y in y_test], pred=y_test_pred),
                    'fit_predict_time': time.time() - start_time
                }
            )
        else:
            raise Exception(f'Undefined reg methods = {method}')

    return pd.DataFrame(res_list)
