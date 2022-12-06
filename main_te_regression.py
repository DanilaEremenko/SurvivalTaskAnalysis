import json
from datetime import timedelta

from pathlib import Path
import pandas as pd
from joblib import load, dump
from pandas.api.types import is_string_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

from lib.drawing import draw_group_bars_and_boxes, draw_corr_sns, draw_pie_chart
from lib.models_building import build_scenarios
from lib.time_ranges import get_time_range_symb


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
    def __init__(self, res_dir: str, train_file: str, test_file: str, y_key: str):
        self.res_dir = Path(res_dir)
        self.train_file = train_file
        self.test_file = test_file
        self.y_key = y_key


if __name__ == '__main__':
    ################################################
    # ------------ exp descriptions  ---------------
    ################################################

    exp_desc = ExpRegDesc(
        res_dir='full_reg_elapsed_time (super fair)',
        train_file='sk-full-data/fair_ds/train.csv',
        test_file='sk-full-data/fair_ds/test.csv',
        y_key='ElapsedRaw'
    )

    exp_desc.res_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(exp_desc.train_file)
    test_df = pd.read_csv(exp_desc.test_file)

    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in train_df.keys()
        if is_string_dtype(train_df[key])
    }

    for key, le in le_dict.items():
        train_df[key] = le.fit_transform(train_df[key])
        test_df[key] = le.fit_transform(test_df[key])

    x_train, y_train = train_df.drop(columns=[exp_desc.y_key]), train_df[exp_desc.y_key]
    x_test, y_test = test_df.drop(columns=[exp_desc.y_key]), test_df[exp_desc.y_key]

    ################################################
    # ------------ search params -------------------
    ################################################
    res_list_df = build_scenarios(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        method='rf',
    )

    res_list_df.sort_values('r', ascending=False, inplace=True)
    res_list_df.to_csv(f'{exp_desc.res_dir}/res_full_search.csv')

    best_args = json.loads(res_list_df.iloc[0]['args_dict'])

    rf = RandomForestRegressor(**best_args)
    rf.fit(X=x_train, y=y_train)

    dump(rf, f'{exp_desc.res_dir}/model_stable.joblib')
    ################################################
    # --- analyze model errors & dependencies  -----
    ################################################
    # rf = RandomForestRegressor(
    #     n_estimators=100,
    #     # max_depth=max_depth,
    #     min_samples_leaf=1,
    #     max_features=0.25,
    #     bootstrap=True,
    #     n_jobs=4,
    #     random_state=42
    # )
    # rf.fit(X=x_train, y=y_train)
    # rf = load(f'{exp_desc.res_dir}/model.joblib')
    #
    # imp_df = pd.DataFrame({'feature': x_test.keys(), 'imp': rf.feature_importances_}) \
    #     .sort_values('imp', ascending=False)
    #
    # y_pred = pd.DataFrame({'y_pred': rf.predict(x_test)})
    # y_pred.to_csv('sk-full-data/fair_ds/y_pred_reg.csv', index=False)
