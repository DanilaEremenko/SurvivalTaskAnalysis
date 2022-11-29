import json
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest


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
            model.fit(X=x_train, y=y_train)
            c_val = model.score(X=x_test[:10_000], y=y_test[:10_000])

            res_list.append(
                {
                    'args_dict': json.dumps(args_dict),
                    'c-val': c_val
                }
            )
        else:
            raise Exception(f'Undefined reg methods = {method}')

    return pd.DataFrame(res_list)
