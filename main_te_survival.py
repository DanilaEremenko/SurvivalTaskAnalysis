import json
import random
import time
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from joblib import dump, load

from experiments_config import EXP_PATH, MODELS_MODE
from lib.custom_survival_funcs import add_events_to_df, batch_surv_time_pred
from lib.losses import Losses
from lib.models_building import build_scenarios


class ExpSurvDesc:
    def __init__(
            self,
            res_dir: str,
            train_file: str,
            test_file: str,
            y_key: str,
            event_key: str,
            translate_func
    ):
        self.res_dir = Path(res_dir)
        self.train_file = train_file
        self.test_file = test_file
        self.y_key = y_key
        self.event_key = event_key
        self.translate_func = translate_func


if __name__ == '__main__':
    ################################################
    # ------------ exp descriptions  ---------------
    ################################################
    exp_desc = ExpSurvDesc(
        res_dir=f"{EXP_PATH}/psearch_surv_elapsed_time",
        train_file=f'{EXP_PATH}/train.csv',
        test_file=f'{EXP_PATH}/test.csv',
        y_key='ElapsedRaw',
        event_key='event',
        translate_func=add_events_to_df
    )

    exp_desc.res_dir.mkdir(exist_ok=True)

    ################################################
    # ------------ data processing  ----------------
    ################################################
    train_df = pd.read_csv(exp_desc.train_file, index_col=0)
    test_df = pd.read_csv(exp_desc.test_file, index_col=0)

    # State is neccessary for event formation
    train_df = exp_desc.translate_func(train_df)
    test_df = exp_desc.translate_func(test_df)

    train_df.drop(columns=['State'], inplace=True)
    test_df.drop(columns=['State'], inplace=True)

    # state_dist = pd.DataFrame([
    #     {
    #         'state': state,
    #         'time': len(src_df[src_df['State'] == state]) / len(src_df),
    #         'ElapsedRawMean': src_df[src_df['State'] == state]['ElapsedRaw'].mean()
    #     }
    #     for state in src_df['State'].unique()]
    # )

    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in train_df.keys()
        if is_string_dtype(train_df[key])
    }

    x_train = train_df.drop(columns=[exp_desc.y_key, exp_desc.event_key])
    y_train_src = train_df[[exp_desc.y_key, exp_desc.event_key]]

    x_test = test_df.drop(columns=[exp_desc.y_key, exp_desc.event_key])
    y_test_src = test_df[[exp_desc.y_key, exp_desc.event_key]]

    for key, le in le_dict.items():
        x_train[key] = le.fit_transform(x_train[key])
        x_test[key] = le.fit_transform(x_test[key])

    # x_all = x_all.iloc[:1_000]
    # y_all_tr = y_all_tr[:1_000]

    y_train = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_train_src)
    y_test = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_test_src)

    ####################################################################################################################
    # ----------------------------------------------- params search | predict ------------------------------------------
    ####################################################################################################################
    if MODELS_MODE == 'search':
        res_list_df = build_scenarios(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            method='rsf'
        )
        res_list_df.sort_values('r2_mean_std', ascending=False, inplace=True)
        res_list_df.to_csv(f'{exp_desc.res_dir}/res_full_search.csv')
    elif MODELS_MODE == 'predict':
        best_params = json.loads(pd.read_csv(f'{exp_desc.res_dir}/res_full_search.csv').iloc[0]['args_dict'])
        rsf = RandomSurvivalForest(**best_params)
        rsf.fit(X=x_train, y=y_train)
        y_pred = pd.DataFrame(
            {
                'y_pred': batch_surv_time_pred(model=rsf, X=x_test,mode='math_exp')
            }
        )
        y_pred.to_csv(f'{EXP_PATH}/y_pred_surv.csv', index=False)
    else:
        raise Exception(f"Unexpected MODELS_MODE = {MODELS_MODE}")
