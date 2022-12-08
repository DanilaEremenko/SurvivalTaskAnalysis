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

from experiments import EXP_PATH
from lib.losses import Losses
from lib.models_building import build_scenarios, get_event_time_manual


def translate_func_simple(df: pd.DataFrame) -> pd.DataFrame:
    # TODO add correct processing for all events
    df.loc[:, 'event'] = 0
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 1
    df.loc[random.choice(df.index), 'event'] = 0
    return df


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
        translate_func=translate_func_simple
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

    # ################################################
    # -------------- search params -------------------
    # ################################################
    # res_list_df = build_scenarios(
    #     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #     method='rsf'
    # )
    #
    # res_list_df.sort_values('r', ascending=False, inplace=True)
    # res_list_df.to_csv(f'{exp_desc.res_dir}/res_full_search.csv')
    # ################################################
    # -------------- deep survival analyze -----------
    # ################################################
    # rsf = load(f'{exp_desc.res_dir}/model.joblib')
    # rsf = RandomSurvivalForest(
    #     n_estimators=54,
    #     bootstrap=True,
    #     max_samples=500,
    #     min_samples_leaf=8,
    #     n_jobs=4,
    #     random_state=42
    # )
    # rsf.fit(X=x_train, y=y_train)
    #
    # y_test_src_sorted = y_test_src.sort_values('ElapsedRaw')
    # # y_test_src_sel = pd.concat([y_test_src_sorted.iloc[:3], y_test_src_sorted.iloc[-3:]])
    # y_test_src_sel = y_test_src_sorted[
    #                      (y_test_src_sorted['ElapsedRaw'] > 1e4) & (y_test_src_sorted['ElapsedRaw'] < 1e5)
    #                      ].iloc[0:10]
    #
    # x_test_sel = x_test.loc[y_test_src_sel.index]
    # # TODO great bone cause Surv.from_dataframe don't allow to pass elements without 0 event
    # saved_event = y_test_src_sel.loc[y_test_src_sel.index[0], 'event']
    # y_test_src_sel.loc[y_test_src_sel.index[0], 'event'] = 0
    # y_test_sel = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_test_src_sel)
    # y_test_sel[0][0] = saved_event
    #
    # surv = rsf.predict_survival_function(x_test_sel, return_array=True)
    # for i, s in enumerate(surv):
    #     plt.step(rsf.event_times_, s, where="post", label=str(i))
    # plt.ylabel("Survival probability")
    # plt.xlabel("Time in seconds")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # imps_raw = permutation_importance(rsf, x_test_sel, y_test_sel, n_repeats=15, random_state=42)
    # imps_df = pd.DataFrame(
    #     {k: imps_raw[k] for k in ("importances_mean", "importances_std",)},
    #     index=x_test_sel.columns
    # ).sort_values(by="importances_mean", ascending=False)
    #
    # start_time = time.time()
    # y_pred = pd.DataFrame(
    #     {
    #         'y_pred': np.concatenate([
    #             get_event_time_manual(probs=rsf.predict_survival_function(x_test[start:start + 200], return_array=True))
    #             for start in range(0, len(x_test), 200)
    #         ])
    #     }
    # )
    # ################################################
    # -------------- made predictions ----------------
    # ################################################
    rsf = RandomSurvivalForest(
        n_estimators=54,
        bootstrap=True,
        max_samples=500,
        min_samples_leaf=8,
        n_jobs=4,
        random_state=42
    )
    rsf.fit(X=x_train, y=y_train)
    y_pred = pd.DataFrame(
        {
            'y_pred': rsf.predict(x_test)
        }
    )
    y_pred.to_csv(f'{EXP_PATH}/y_pred_surv.csv', index=False)
