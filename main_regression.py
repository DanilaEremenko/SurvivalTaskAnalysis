import json
from datetime import timedelta

from pathlib import Path
import pandas as pd
from pandas.api.types import is_string_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

from drawing import draw_group_bars_and_boxes, draw_corr_sns, draw_pie_chart
from models_building import get_reg_predictions_and_metrics_df, build_scenarios
from time_ranges import get_time_range_symb


def draw_bars_and_boxes_by_categories(df: pd.DataFrame, group_keys: List[str], y_key: str, res_dir: Path):
    res_dir.mkdir(exist_ok=True, parents=True)

    for group_key in group_keys:
        res_path = f'{res_dir}/{y_key}_agg_by_{group_key.replace(":", "")}.html'
        print(f'drawing {res_path}')
        draw_group_bars_and_boxes(df=df, group_key=group_key, y_key=y_key, res_path=res_path)


def draw_correlations(res_test_df: pd.DataFrame, res_dir: Path):
    draw_corr_sns(
        group_df=res_test_df,
        x_key='y_true', x_title='True time (seconds)',
        y_key='y_pred', y_title='Estimated time (seconds)',
        add_mae=False, add_rmse=False, add_mae_perc=True,
        title='All correlation',
        kind='reg',
        res_dir=res_dir,
    )

    for time_range in res_test_df['time_range'].unique():
        draw_corr_sns(
            group_df=res_test_df[res_test_df['time_range'] == time_range],
            x_key='y_true', x_title='True time (seconds)',
            y_key='y_pred', y_title='Estimated time (seconds)',
            add_mae=False, add_rmse=False, add_mae_perc=True,
            title=f'{time_range} correlation',
            kind='reg',
            res_dir=res_dir.joinpath(time_range),
        )


class ExpRegDesc:
    def __init__(self, res_dir: str, src_file: str, y_key: str, ignored_keys: List[str]):
        self.res_dir = Path(res_dir)
        self.src_file = src_file
        self.y_key = y_key
        self.ignored_keys = ignored_keys


if __name__ == '__main__':
    ################################################
    # ------------ exp descriptions  ---------------
    ################################################
    exp_list = [
        ExpRegDesc(res_dir='full_reg_cpu_time', src_file='sk-data/full_data.csv', y_key='CPUTimeRAW',
                   ignored_keys=['ElapsedRaw', 'ElapsedRaw_mean', 'ElapsedRawClass', 'Unnamed: 0']),

        ExpRegDesc(res_dir='full_reg_elapsed_time (super fair)', src_file='sk-data/full_data.csv',
                   y_key='ElapsedRaw',
                   ignored_keys=[
                       'CPUTimeRAW', 'ElapsedRaw_mean', 'ElapsedRaw_std', 'ElapsedRawClass',
                       'SizeClass',
                       'AveCPU', 'AveCPU_mean', 'AveCPU_std',
                       'Unnamed: 0', 'State', 'MinMemoryNode'
                   ]),

    ]

    exp_desc = exp_list[1]
    exp_desc.res_dir.mkdir(exist_ok=True)

    ################################################
    # ------------ data processing  ----------------
    ################################################
    src_df = pd.read_csv(exp_desc.src_file)

    corr_df = src_df.corr(numeric_only=True)

    # get most correlated with target value columns
    # corr_df[exp_desc.y_key].sort_values()

    filt_df = src_df.dropna(axis=1)

    filt_ignored_keys = [key for key in exp_desc.ignored_keys if key in filt_df.keys()]
    filt_df = filt_df.drop(columns=filt_ignored_keys)

    filt_df = filt_df[filt_df[exp_desc.y_key] != 0]

    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }

    x_all, y_all = filt_df.drop(columns=[exp_desc.y_key]), filt_df[exp_desc.y_key]

    for key, le in le_dict.items():
        x_all[key] = le.fit_transform(x_all[key])

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.33,
        # shuffle=True
    )

    ################################################
    # ------------ building models -----------------
    ################################################
    res_list_df = build_scenarios(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        method='rf',
        args_scenarios=[
            dict(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                max_features=max_features,
                n_jobs=4,
                random_state=42
            )
            # # search
            # for n_estimators in [10, 50, 150]
            # for min_samples_leaf in [1, 2, 4]
            # for bootstrap in [True, False]
            # for max_features in [1.0, 0.75, 0.5, 0.25]

            # stable
            for n_estimators in [50]
            for min_samples_leaf in [1]
            for bootstrap in [False]
            for max_features in [0.25]
        ]
    )

    for key, le in le_dict.items():
        res_list_df[key] = le.inverse_transform(res_list_df[key])

    res_list_df.sort_values('mae', inplace=True)
    res_list_df.to_csv(f'{exp_desc.res_dir}/res_last_search.csv')

    ################################################
    # --- analyze model errors & dependencies  -----
    ################################################
    best_args = json.loads(res_list_df.iloc[0]['args_dict'])

    rf = RandomForestRegressor(**best_args)
    rf.fit(X=x_train, y=y_train)

    imp_df = pd.DataFrame({'feature': x_test.keys(), 'imp': rf.feature_importances_}).sort_values('imp')

    res_test_df = get_reg_predictions_and_metrics_df(model=rf, X=x_test, y=y_test)

    for key, le in le_dict.items():
        res_test_df[key] = le.inverse_transform(res_test_df[key])

    res_test_df['time_range'] = [get_time_range_symb(task_time=task_time)
                                 for task_time in list(res_test_df['y_true'])]

    src_df['time_range'] = [get_time_range_symb(task_time=task_time)
                            for task_time in list(src_df[exp_desc.y_key])]

    # bars & boxes
    draw_bars_and_boxes_by_categories(df=src_df, group_keys=list(le_dict.keys()), y_key=exp_desc.y_key,
                                      res_dir=exp_desc.res_dir.joinpath('dist').joinpath('y_by_categories'))
    draw_bars_and_boxes_by_categories(df=res_test_df, group_keys=list(le_dict.keys()), y_key='mae perc',
                                      res_dir=exp_desc.res_dir.joinpath('dist').joinpath('test_error_by_categories'))
    draw_bars_and_boxes_by_categories(df=res_test_df, group_keys=list(le_dict.keys()), y_key='y_pred',
                                      res_dir=exp_desc.res_dir.joinpath('dist').joinpath('test_pred_by_categories'))
    # correlation
    draw_correlations(res_test_df=res_test_df, res_dir=exp_desc.res_dir.joinpath('corr'))

    # pie chart dist
    for pie_mode in ['sum', 'count']:
        draw_pie_chart(
            df=src_df,
            group_key='time_range', sum_key=exp_desc.y_key,
            mode=pie_mode,
            res_path=f"{exp_desc.res_dir}/range_dist_mode={pie_mode}.html"
        )
