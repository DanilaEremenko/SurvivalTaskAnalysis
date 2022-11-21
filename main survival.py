import time
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from joblib import dump, load

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
            res_dir='full_surv_elapsed_time (simple super fair)', src_file='sk-full-data/full_data.csv',
            y_key='ElapsedRaw',
            event_key='event',
            translate_func=translate_func_simple,
            ignored_keys=[
                'CPUTimeRAW', 'ElapsedRaw_mean', 'ElapsedRaw_std', 'ElapsedRawClass',
                'SizeClass',
                'AveCPU', 'AveCPU_mean', 'AveCPU_std',
                'Unnamed: 0', 'State', 'MinMemoryNode'
            ]
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

    filt_ignored_keys = [key for key in exp_desc.ignored_keys if key in filt_df.keys()]
    filt_df = filt_df.drop(columns=filt_ignored_keys)

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

    # x_all = x_all.iloc[:1_000]
    # y_all_tr = y_all_tr[:1_000]

    x_train, x_test, y_train_src, y_test_src = train_test_split(
        x_all, y_all, test_size=0.40,
        # shuffle=True
    )

    y_train = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_train_src)
    y_test = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_test_src)

    # ################################################
    # -------------- search params -------------------
    # ################################################
    # res_list_df = build_scenarios(
    #     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #     method='rsf',
    #     args_scenarios=[
    #         dict(
    #             n_estimators=n_estimators,
    #             min_samples_leaf=min_samples_leaf,
    #             bootstrap=bootstrap,
    #             max_features=max_features,
    #             n_jobs=4,
    #             random_state=42
    #         )
    #         # # search
    #         for n_estimators in [10, 50, 150]
    #         for min_samples_leaf in [1, 2, 4]
    #         for bootstrap in [True, False]
    #         for max_features in [1.0, 0.75, 0.5, 0.25]
    #
    #         # stable
    #         # for n_estimators in [100]
    #         # for min_samples_leaf in [4]
    #         # for bootstrap in [True]
    #         # for max_features in [1.]
    #     ]
    # )
    #
    # for key, le in le_dict.items():
    #     res_list_df[key] = le.inverse_transform(res_list_df[key])
    #
    # res_list_df.sort_values('c-val', inplace=True)
    # res_list_df.to_csv(f'{exp_desc.res_dir}/res_last_search.csv')

    # ################################################
    # -------------- fit minimal params --------------
    # ################################################
    # rf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=4, bootstrap=True, max_features=1.)
    # rsf = RandomSurvivalForest(
    #     n_estimators=10,
    #     min_samples_leaf=4,
    #     bootstrap=True,
    #     max_features=1.,
    #     max_samples=10_000,
    #     random_state=4, n_jobs=4
    # )
    # print("fit best params..")
    #
    # start_fit_time = time.time()
    # rsf.fit(X=x_train, y=y_train)
    # print(f'fit time = {time.time() - start_fit_time}')
    #
    # score_time_list = []
    # for samples_num in [2500, 5000, 10_000]:
    #     start_score_time = time.time()
    #     score = rsf.score(X=x_test.iloc[0:samples_num], y=y_test[0:samples_num])
    #     score_time_list.append(
    #         {
    #             'score': score,
    #             'samples_num': samples_num,
    #             'time': time.time() - start_score_time
    #         }
    #     )
    #     print(score_time_list[-1])
    #
    # score_time_df = pd.DataFrame(score_time_list)
    #
    # dump(rsf, f'{exp_desc.res_dir}/model.joblib')

    # ################################################
    # -------------- upload model --------------------
    # ################################################
    rsf = load(f'{exp_desc.res_dir}/model.joblib')

    y_test_src_sorted = y_test_src.sort_values('ElapsedRaw')
    # y_test_src_sel = pd.concat([y_test_src_sorted.iloc[:3], y_test_src_sorted.iloc[-3:]])
    y_test_src_sel = y_test_src_sorted[
                         (y_test_src_sorted['ElapsedRaw'] > 1e4) & (y_test_src_sorted['ElapsedRaw'] < 1e5)
                         ].iloc[0:10]

    x_test_sel = x_test.loc[y_test_src_sel.index]
    y_test_src_sel.loc[0, 'event'] = 0  # TODO great bone
    y_test_sel = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_test_src_sel)

    surv = rsf.predict_survival_function(x_test_sel, return_array=True)
    for i, s in enumerate(surv):
        plt.step(rsf.event_times_, s, where="post", label=str(i))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in seconds")
    plt.legend()
    plt.grid(True)
    plt.show()

    imps_raw = permutation_importance(rsf, x_test_sel, y_test_sel, n_repeats=15, random_state=42)
    imps_df = pd.DataFrame(
        {k: imps_raw[k] for k in ("importances_mean", "importances_std",)},
        index=x_test_sel.columns
    ).sort_values(by="importances_mean", ascending=False)
