from pathlib import Path
from typing import List, Dict

import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sksurv.util import Surv

from models_building import build_scenarios


def translate_func_simple(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['State'].isin(['COMPLETED', 'TIMEOUT', 'RUNNING'])].copy()
    df.loc[:, 'event'] = 0
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 1
    return df


class ExpSurvDesc:
    def __init__(
            self,
            res_dir: str,
            src_file: str,
            y_key: str,
            event_key: str,
            translate_func,
            ignored_keys: List[str]
    ):
        self.res_dir = Path(res_dir)
        self.src_file = src_file
        self.y_key = y_key
        self.event_key = event_key
        self.translate_func = translate_func
        self.ignored_keys = ignored_keys


if __name__ == '__main__':
    ################################################
    # ------------ exp descriptions  ---------------
    ################################################
    exp_list = [
        ExpSurvDesc(
            res_dir='full_surv_elapsed_time (simple)', src_file='sk-full-data/full_data.csv',
            y_key='ElapsedRaw',
            event_key='event',
            translate_func=translate_func_simple,
            ignored_keys=['CPUTimeRAW', 'ElapsedRaw_mean', 'Unnamed: 0', 'State', 'MinMemoryNode']
        )
    ]

    exp_desc = exp_list[0]
    exp_desc.res_dir.mkdir(exist_ok=True)

    ################################################
    # ------------ data processing  ----------------
    ################################################
    src_df = pd.read_csv(exp_desc.src_file)
    filt_df = exp_desc.translate_func(src_df)

    # state_dist = pd.DataFrame([
    #     {
    #         'state': state,
    #         'time': len(src_df[src_df['State'] == state]) / len(src_df),
    #         'ElapsedRawMean': src_df[src_df['State'] == state]['ElapsedRaw'].mean()
    #     }
    #     for state in src_df['State'].unique()]
    # )

    corr_df = src_df.corr(numeric_only=True)

    # get most correlated with target value columns
    # corr_df[exp_desc.y_key].sort_values()

    filt_df = filt_df.dropna(axis=1)
    filt_df = filt_df.drop(columns=exp_desc.ignored_keys)

    filt_df = filt_df[filt_df[exp_desc.y_key] != 0]

    # # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }

    x_all = filt_df.drop(columns=[exp_desc.y_key, exp_desc.event_key])
    y_all = filt_df[[exp_desc.y_key, exp_desc.event_key]]

    for key, le in le_dict.items():
        x_all[key] = le.fit_transform(x_all[key])

    y_all_tr = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_all)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all_tr, test_size=0.33,
        # shuffle=True
    )

    # ################################################
    # # ------------ building models -----------------
    # ################################################
    res_list_df = build_scenarios(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        method='rsf',
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
            for n_estimators in [10, 50, 150]
            for min_samples_leaf in [1, 2, 4]
            for bootstrap in [True, False]
            for max_features in [1.0, 0.75, 0.5, 0.25]

            # stable
            # for n_estimators in [50]
            # for min_samples_leaf in [1]
            # for bootstrap in [False]
            # for max_features in [0.25]
        ]
    )

    for key, le in le_dict.items():
        res_list_df[key] = le.inverse_transform(res_list_df[key])

    res_list_df.sort_values('c-val', inplace=True)
    res_list_df.to_csv(f'{exp_desc.res_dir}/res_last_search.csv')
