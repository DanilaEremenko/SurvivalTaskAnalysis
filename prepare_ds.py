from pathlib import Path
from typing import List, Dict

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

filt_df = pd.read_csv('sk-full-data/last_data/data.csv', index_col=0)
areas_df = pd.read_csv('sk-full-data/last_data/areas.csv')
res_dir = Path('sk-full-data/fair_ds')
res_dir.mkdir(exist_ok=True)

print(f"filter geovation tasks {len(filt_df[filt_df['GroupID_scontrol'] != 'geovation(50218)']) / len(filt_df)}")
filt_df = filt_df[filt_df['GroupID_scontrol'] != 'geovation(50218)']

ignored_keys = [
    'UserID', 'GroupID', 'JobID',
    'CPUTimeRAW', 'ElapsedRaw_mean', 'ElapsedRaw_std', 'ElapsedRawClass',
    'SizeClass',
    'AveCPU', 'AveCPU_mean', 'AveCPU_std',
    'Unnamed: 0', 'State', 'MinMemoryNode',

    'SuspendTime', 'ReqNodeList', 'ExcNodeList', 'Features',
    'Dependency',

]

ignored_keys = [key for key in ignored_keys if key in filt_df.keys()]
filt_df.dropna(axis=1, inplace=True)


def drop_group_of_keys(drop_keys: List[str], reason: str):
    print(f"cause of {reason} drop: {','.join(drop_keys)}")
    filt_df.drop(columns=drop_keys, inplace=True)


drop_group_of_keys([key for key in filt_df.keys() if len(filt_df[key].unique()) > 1000 and key != 'ElapsedRaw'],
                   '> 1000 different values')
drop_group_of_keys([key for key in filt_df.keys() if len(filt_df[key].unique()) == 1],
                   'only 1 different value')
drop_group_of_keys([key for key in ignored_keys if key in filt_df.keys()],
                   'manually defined keys')


def get_high_corr_keys() -> List[str]:
    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if not is_numeric_dtype(filt_df[key])
    }

    for key, le in le_dict.items():
        filt_df[key] = [str(val) for val in list(filt_df[key])]
        filt_df[key] = le.fit_transform(filt_df[key])

    corr_df = filt_df.corr(numeric_only=True)
    high_corr_keys = []
    for key1 in filt_df.keys():
        for key2 in filt_df.keys():
            if key2 != key1 and \
                    key1 not in high_corr_keys and \
                    key2 not in high_corr_keys and \
                    corr_df.loc[key1, key2] > 0.9:
                high_corr_keys.append(key2)

    for key, le in le_dict.items():
        filt_df[key] = le.inverse_transform(filt_df[key])

    return high_corr_keys


drop_group_of_keys(get_high_corr_keys(), 'high correlation with another key')


def low_rf_importance_keys():
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if not is_numeric_dtype(filt_df[key])
    }

    for key, le in le_dict.items():
        filt_df[key] = [str(val) for val in list(filt_df[key])]
        filt_df[key] = le.fit_transform(filt_df[key])

    rf = RandomForestRegressor(n_estimators=50, max_depth=4, bootstrap=False)
    x_keys = [key for key in filt_df.keys() if key != 'ElapsedRaw']
    rf.fit(
        X=filt_df[x_keys],
        y=filt_df['ElapsedRaw']
    )
    imp_df = pd.DataFrame([{'key': key, 'imp': imp} for key, imp in zip(x_keys, rf.feature_importances_)]) \
        .sort_values('imp', ascending=False)

    for key, le in le_dict.items():
        filt_df[key] = le.inverse_transform(filt_df[key])

    return list(imp_df[imp_df['imp'] < 0.001]['key'])


drop_group_of_keys(low_rf_importance_keys(), 'low importance in random forrest')

# filt_df = filt_df[filt_df['ElapsedRaw'] > 0]
# filt_df = filt_df[[key for key in filt_df.keys() if 'License' not in key]]

train_df, test_df = train_test_split(filt_df, test_size=0.33, random_state=42)

# useful in debug
# {group: len(filt_df[filt_df['GroupID_scontrol'] == group]) / len(filt_df)
#  for group in filt_df['GroupID_scontrol'].unique()}
# {key: len(filt_df[key].unique()) for key in filt_df.keys()}
# {key: filt_df[key].unique() for key in filt_df.keys()}


train_df.to_csv(f'{res_dir}/train.csv')
test_df.to_csv(f'{res_dir}/test.csv')
