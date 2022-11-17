import json
import time
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from random_survival_forest.models import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index


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


def get_surv_c_val(model, X, y) -> float:
    y_pred = model.predict(X=X)
    c_val = concordance_index(y_time=y["ElapsedRaw"], y_pred=y_pred, y_event=y["event"])
    return c_val


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
                    'mae': mae
                }
            )

        elif method == 'rsf':
            model = RandomSurvivalForest(**args_dict)

            start_time = time.time()
            model.fit(x=x_train, y=y_train)
            print(f"rsf fit time   = {time.time() - start_time}")

            start_time = time.time()
            c_val = get_surv_c_val(model=model, X=x_test, y=y_test)
            print(f"rsf c-val time = {time.time() - start_time}")

            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'c-val': c_val
                }
            )
        else:
            raise Exception(f'Undefined reg methods = {method}')

    return pd.DataFrame(res_list)
