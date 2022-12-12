import json
from datetime import timedelta

from pathlib import Path
import pandas as pd
from joblib import load, dump
from lightgbm import LGBMRegressor
from pandas.api.types import is_string_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

from experiments_config import EXP_PATH, MODELS_MODE
from lib.drawing import draw_group_bars_and_boxes, draw_corr_sns, draw_pie_chart
from lib.models_building import build_scenarios, assym_obj_fn, assym_valid_fn
from lib.time_ranges import get_time_range_symb


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
    reg_method = 'rf'
    exp_desc = ExpRegDesc(
        res_dir=f"{EXP_PATH}/psearch_reg_{reg_method}_elapsed_time",
        train_file=f'{EXP_PATH}/train.csv',
        test_file=f'{EXP_PATH}/test.csv',
        y_key='ElapsedRaw'
    )

    exp_desc.res_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(exp_desc.train_file)
    test_df = pd.read_csv(exp_desc.test_file)

    train_df.drop(columns=['State'], inplace=True)
    test_df.drop(columns=['State'], inplace=True)

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

    ####################################################################################################################
    # ----------------------------------------------- params search | predict ------------------------------------------
    ####################################################################################################################
    if MODELS_MODE == 'search':
        res_list_df = build_scenarios(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            method=reg_method,
        )
        res_list_df.sort_values('r', ascending=False, inplace=True)
        res_list_df.to_csv(f'{exp_desc.res_dir}/res_full_search.csv')
    elif MODELS_MODE == 'predict':
        best_params = json.loads(pd.read_csv(f'{exp_desc.res_dir}/res_full_search.csv').iloc[0]['args_dict'])
        if reg_method == 'rf':
            model = RandomForestRegressor(**best_params)
            model.fit(X=x_train.to_numpy(), y=y_train)
        elif reg_method == 'lgbm':
            model = LGBMRegressor(**best_params)
            model.set_params(objective=assym_obj_fn)
            model.fit(X=x_train.to_numpy(), y=y_train, eval_metric=assym_valid_fn)
        else:
            raise Exception(f"Unexpected reg method = {reg_method}")

        imp_df = pd.DataFrame({'feature': x_test.keys(), 'imp': model.feature_importances_}) \
            .sort_values('imp', ascending=False)

        y_pred = pd.DataFrame({'y_pred': model.predict(x_test)})
        y_pred.to_csv(f'{EXP_PATH}/y_pred_reg_{reg_method}.csv', index=False)
    else:
        raise Exception(f"Unexpected MODELS_MODE = {MODELS_MODE}")
