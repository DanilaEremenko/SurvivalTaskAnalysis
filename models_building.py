import json
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def get_predictions_and_residuals_df(model, X, y) -> pd.DataFrame:
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
        le_dict: Dict[str, LabelEncoder],
        args_scenarios: List[dict],
        reg_method: str
) -> pd.DataFrame:
    res_list = []

    for i, args_dict in enumerate(args_scenarios):
        print(f'fitting scenario {i}/{len(args_scenarios)}')

        if reg_method == 'rf':
            rf = RandomForestRegressor(**args_dict)
        else:
            raise Exception(f'Undefined reg methods = {reg_method}')

        rf.fit(X=x_train, y=y_train)

        res_test_df = get_predictions_and_residuals_df(model=rf, X=x_test, y=y_test)

        for key, le in le_dict.items():
            res_test_df[key] = le.inverse_transform(res_test_df[key])

        mae = res_test_df['mae perc'].mean()

        res_list.append(
            {
                'args_dict': json.dumps(args_dict),
                'mae': mae
            }
        )

    res_list_df = pd.DataFrame(res_list).sort_values('mae')

    return res_list_df
