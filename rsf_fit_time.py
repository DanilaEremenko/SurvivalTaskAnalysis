import time
from typing import Dict

import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def translate_func_simple(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['State'].isin(['COMPLETED', 'TIMEOUT', 'RUNNING'])].copy()
    df.loc[:, 'event'] = 0
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 1
    return df


src_df = pd.read_csv('sk-full-data/full_data.csv')
filt_df = translate_func_simple(src_df)

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
filt_df = filt_df.drop(columns=['CPUTimeRAW', 'ElapsedRaw_mean', 'Unnamed: 0', 'State', 'MinMemoryNode'])

filt_df = filt_df[filt_df['ElapsedRaw'] != 0]

# # initialize label encoders
le_dict: Dict[str, LabelEncoder] = {
    key: LabelEncoder() for key in filt_df.keys()
    if is_string_dtype(filt_df[key])
}

x_all = filt_df.drop(columns=['ElapsedRaw', 'event'])
y_all = filt_df[['ElapsedRaw', 'event']]

# remove columns
# x_all = x_all[[key for key in list(x_all.keys())[:20]]]

for key, le in le_dict.items():
    x_all[key] = le.fit_transform(x_all[key])

y_all_tr = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_all)

res_list = []

rf = RandomSurvivalForest(n_estimators=100)

for samples_num in [10, 100, 1000, 10_000]:
    # samples_num = 100
    # for n_esimators in [10, 100, 200]:
    fit_start_time = time.time()
    rf.fit(X=x_all.iloc[:samples_num], y=y_all_tr[:samples_num])
    fit_end_time = time.time()
    score = rf.score(X=x_all.iloc[:samples_num], y=y_all_tr[:samples_num])
    score_end_time = time.time()

    res_list.append(
        {
            'samples_num': samples_num,
            'score': round(score, 2),
            'fit_time': fit_end_time - fit_start_time,
            'score_time': score_end_time - fit_end_time,
        }
    )

    print(res_list[-1])

res_df = pd.DataFrame(res_list)
