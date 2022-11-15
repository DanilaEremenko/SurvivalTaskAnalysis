import json
import random

from pathlib import Path
import pandas as pd
from pandas.api.types import is_string_dtype

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List


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


def get_random_color() -> str:
    return "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])


def draw_group_agg_info(df: pd.DataFrame, group_key: str, y_key: str, res_path: str):
    random.seed(42)
    row_num = 2
    fig = make_subplots(
        rows=row_num, cols=1,
        vertical_spacing=0.5 / row_num
    )
    for group_val in df[group_key].unique():
        curr_df = df[df[group_key] == group_val]
        curr_color = get_random_color()
        fig.add_trace(
            go.Bar(
                name=group_val,
                legendgroup=group_val,
                x=[group_val],
                y=[len(curr_df)],
                marker_color=curr_color
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(
                name=group_val,
                legendgroup=group_val,
                y=curr_df[y_key],
                marker_color=curr_color
            ),
            row=2, col=1
        )

    fig.update_xaxes(tickangle=30)
    fig.update_layout(title_text=f'mae percentage error grouped by {group_key}')

    fig.write_html(res_path)
    # fig.show()


def draw_dependecies(df: pd.DataFrame, group_keys: List[str], y_key: str, res_dir: str):
    Path(res_dir).mkdir(exist_ok=True)

    for group_key in group_keys:
        res_path = f'{res_dir}/{y_key}_agg_by_{group_key.replace(":", "")}.html'
        print(f'drawing {res_path}')
        draw_group_agg_info(df=df, group_key=group_key, y_key=y_key, res_path=res_path)


class ExpDesc:
    def __init__(self, src_file: str, y_key: str, ignored_keys: List[str]):
        self.src_file = src_file
        self.y_key = y_key
        self.ignored_keys = ignored_keys


if __name__ == '__main__':
    ################################################
    # ------------ data processing  ----------------
    ################################################
    exp_list = [
        ExpDesc(src_file='sk-data/full_data.csv', y_key='CPUTimeRAW',
                ignored_keys=['ElapsedRaw', 'ElapsedRaw_mean', 'ElapsedRawClass', 'Unnamed: 0']),
        ExpDesc(src_file='sk-data/data.csv', y_key='ElapsedRaw',
                ignored_keys=['ElapsedRaw_mean', 'ElapsedRawClass', 'Unnamed: 0'])

    ]

    exp_desc = exp_list[0]

    src_df = pd.read_csv(exp_desc.src_file)

    corr_df = src_df.corr(numeric_only=True)

    # get most correlated with target value columns
    # corr_df[exp_desc.y_key].sort_values()

    filt_df = src_df.dropna(axis=1)
    filt_df = filt_df.drop(columns=exp_desc.ignored_keys)

    filt_df = filt_df[filt_df[exp_desc.y_key] != 0]

    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }

    x_all, y_all = filt_df.drop(columns=[exp_desc.y_key]), filt_df[exp_desc.y_key]

    for key, le in le_dict.items():
        x_all[key] = le.fit_transform(x_all[key])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.33, shuffle=True)

    ################################################
    # ------------ building models -----------------
    ################################################
    res_list = []

    args_scenarios = [
        dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=4,
            random_state=42
        )
        # # search
        # for n_estimators in [10, 50, 150]
        # for min_samples_leaf in [1, 2, 4]
        # for bootstrap in [True, False]

        # stable
        for n_estimators in [50]
        for min_samples_leaf in [1]
        for bootstrap in [False]
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

    res_list_df = pd.DataFrame(res_list).sort_values('mae')

    ################################################
    # --- analyze model errors & dependencies  -----
    ################################################
    best_args = json.loads(res_list_df.iloc[0]['args_dict'])

    rf = RandomForestRegressor(**best_args)
    rf.fit(X=x_train, y=y_train)

    imp_df = pd.DataFrame({'feature': x_test.keys(), 'imp': rf.feature_importances_}).sort_values('imp')

    res_test_df = get_res_test_df(model=rf, x_test=x_test, y_test=y_test)

    for key, le in le_dict.items():
        res_test_df[key] = le.inverse_transform(res_test_df[key])

    draw_dependecies(df=src_df, group_keys=list(le_dict.keys()), y_key='TimelimitRaw', res_dir='src_deps')
    draw_dependecies(df=res_test_df, group_keys=list(le_dict.keys()), y_key='mae perc', res_dir='test_error_deps')
