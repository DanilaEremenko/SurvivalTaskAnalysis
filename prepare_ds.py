from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

########################################################################################################################
# ------------------------------------ loading & merging ---------------------------------------------------------------
########################################################################################################################
# load
filt_df = pd.read_csv('sk-full-data/last_data/data.csv', index_col=0)
areas_df = pd.read_csv('sk-full-data/last_data/areas.csv')
res_dir = Path('sk-full-data/fair_ds_nogeov')
res_dir.mkdir(exist_ok=True)

# geovation drop | save
print(f"filter geovation tasks {len(filt_df[filt_df['GroupID_scontrol'] == 'geovation(50218)']) / len(filt_df)}")
filt_df = filt_df[filt_df['GroupID_scontrol'] != 'geovation(50218)']

# from lib.time_ranges import get_time_range_symb
# filt_df.loc[:, 'time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
#                                         for task_time in list(filt_df['ElapsedRaw'])]
#
# {tr:len(filt_df[filt_df['time_elapsed_range']==tr])/len(filt_df) for tr in filt_df['time_elapsed_range'].unique()}

# domains merge
filt_df['GroupID'] = [group_trash.split('(')[0] for group_trash in filt_df['GroupID_scontrol']]
filt_df = pd.merge(left=filt_df, right=areas_df, left_on='GroupID', right_on='GroupID')

filt_df = filt_df[sorted(filt_df.keys())]
########################################################################################################################
# ------------------------------------ condition based & mannually defined drops ---------------------------------------
########################################################################################################################
old_keys = filt_df.keys()
filt_df.dropna(axis=1, inplace=True)
print(f"dropna columns = {','.join([key for key in filt_df if key not in old_keys])}")


def drop_group_of_keys(drop_keys: List[str], reason: str):
    drop_keys = [key for key in drop_keys if key in filt_df.keys()]
    print(f"cause of {reason} drop: {','.join(drop_keys)}")
    filt_df.drop(columns=drop_keys, inplace=True)


manually_defined = [
    # 'UserID', 'GroupID', 'JobID',
    # 'CPUTimeRAW', 'ElapsedRaw_mean', 'ElapsedRaw_std', 'ElapsedRawClass',
    # 'SizeClass',
    # 'AveCPU', 'AveCPU_mean', 'AveCPU_std',
    # 'Unnamed: 0', 'State', 'MinMemoryNode',
    #
    # 'SuspendTime', 'ReqNodeList', 'ExcNodeList', 'Features',
    # 'Dependency',
]
drop_group_of_keys([key for key in manually_defined if key in filt_df.keys()],
                   'manually defined keys')
drop_group_of_keys([key for key in filt_df.keys() if len(filt_df[key].unique()) > 1000 and key != 'ElapsedRaw'],
                   '> 1000 different values')
drop_group_of_keys([key for key in filt_df.keys() if len(filt_df[key].unique()) == 1],
                   'only 1 different value')
########################################################################################################################
# ------------------------------------ transform categorical data & normalize ------------------------------------------
########################################################################################################################
le_dict: Dict[str, LabelEncoder] = {
    key: LabelEncoder() for key in filt_df.keys()
    if not is_numeric_dtype(filt_df[key])
}

for key, le in le_dict.items():
    filt_df[key] = [str(val) for val in list(filt_df[key])]
    filt_df[key] = le.fit_transform(filt_df[key])


########################################################################################################################
# ------------------------------------ correlation & rf importance filtration ------------------------------------------
########################################################################################################################
def get_high_corr_keys() -> List[str]:
    corr_df = filt_df.corr(numeric_only=True)
    high_corr_keys = []
    for key1 in filt_df.keys():
        for key2 in filt_df.keys():
            if key2 != key1 and \
                    key1 not in high_corr_keys and \
                    key2 not in high_corr_keys and \
                    key2 != 'ElapsedRaw' and \
                    corr_df.loc[key1, key2] > 0.9:
                high_corr_keys.append(key2)

    return high_corr_keys


drop_group_of_keys(get_high_corr_keys(), 'high correlation with another key')


def low_rf_importance_keys():
    rf = RandomForestRegressor(n_estimators=50, max_depth=4, bootstrap=False)
    x_keys = [key for key in filt_df.keys() if key != 'ElapsedRaw']
    rf.fit(
        X=filt_df[x_keys],
        y=filt_df['ElapsedRaw']
    )
    imp_df = pd.DataFrame([{'key': key, 'imp': imp} for key, imp in zip(x_keys, rf.feature_importances_)]) \
        .sort_values('imp', ascending=False)

    print(f"top importance = {imp_df[imp_df['imp'] > 0.001][['key', 'imp']]}")

    return list(imp_df[imp_df['imp'] <= 0.001]['key'])


drop_group_of_keys(low_rf_importance_keys(), 'low importance in random forrest')

########################################################################################################################
# ------------------------------- inverse categorical data & normalize transform ---------------------------------------
########################################################################################################################
# le_dict = {key: val for key, val in le_dict.items() if key in filt_df.keys()}
# for key, le in le_dict.items():
#     filt_df[key] = le.inverse_transform(filt_df[key])

filt_df['State'] = filt_df['State'].astype(dtype=int)
filt_df['State'] = le_dict['State'].inverse_transform(filt_df['State'])
########################################################################################################################
# ------------------------------------------- split & save  ------------------------------------------------------------
########################################################################################################################
train_df, test_df = train_test_split(filt_df, test_size=0.33, random_state=42)

train_df.to_csv(f'{res_dir}/train.csv')
test_df.to_csv(f'{res_dir}/test.csv')

# useful in debug
# {group: len(filt_df[filt_df['GroupID_scontrol'] == group]) / len(filt_df)
#  for group in filt_df['GroupID_scontrol'].unique()}
# {key: len(filt_df[key].unique()) for key in filt_df.keys()}
# {key: filt_df[key].unique() for key in filt_df.keys()}
