import json
from typing import Dict

import pandas as pd
from pandas.api.types import is_string_dtype

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_res_test_df(model, x_test, y_test) -> pd.DataFrame:
    res_test_df = pd.DataFrame(
        {
            'y_true': y_test,
            'y_pred': model.predict(X=x_test),
        }
    )

    res_test_df = pd.merge(left=x_test, right=res_test_df, left_index=True, right_index=True)

    res_test_df['residual'] = res_test_df['y_true'] - res_test_df['y_pred']
    res_test_df['residual perc'] = res_test_df['residual'] / res_test_df['y_true'] * 100.

    res_test_df['mae'] = res_test_df['residual'].abs()
    res_test_df['mae perc'] = res_test_df['residual perc'].abs()
    return res_test_df


if __name__ == '__main__':
    df = pd.read_csv('sk-data/full_data.csv')

    corr_df = df.corr(numeric_only=True)
    # corr_df['ElapsedRaw'].sort_values()

    filt_df = df.dropna(axis=1)
    filt_df = filt_df.drop(columns=['ElapsedRaw', 'ElapsedRaw_mean', 'ElapsedRawClass'])
    filt_df = filt_df[filt_df['CPUTimeRAW'] != 0]

    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }

    x_all, y_all = filt_df.drop(columns=['CPUTimeRAW']), filt_df['CPUTimeRAW']

    for key, le in le_dict.items():
        x_all[key] = le.fit_transform(x_all[key])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.33)

    res_list = []

    args_scenarios = [
        dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=4,
            random_state=42
        )
        for n_estimators in [10, 100, 500]
        for min_samples_leaf in [1, 2, 4, 8]
        for bootstrap in [True, False]
    ]

    for i, args_dict in enumerate(args_scenarios):
        print(f'fitting scenario {i}/{len(args_scenarios)}')

        rf = RandomForestRegressor(**args_dict)
        rf.fit(X=x_train, y=y_train)

        res_test_df = get_res_test_df(model=rf, x_test=x_test, y_test=y_test)

        for key, le in le_dict.items():
            res_test_df[key] = le.inverse_transform(res_test_df[key])

        mae = res_test_df['mae perc'].mean()

        res_list.append(
            {
                'args_dict': json.dumps(args_dict),
                'mae': mae
            }
        )
