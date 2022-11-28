import time
from typing import Dict

import numpy as np
from pandas.core.dtypes.common import is_string_dtype
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from time_ranges import get_time_range_symb

df = pd.read_csv('sk-full-data/full_data.csv')
features_keys = ['NumTasks', 'CPUs/Task', 'Priority', 'TimelimitRaw']
y_key = 'ElapsedRaw'
# filt_df = df[[*features_keys, y_key]]
filt_df = df.drop(columns=['MinMemoryNode']).dropna(axis=1)

le_dict: Dict[str, LabelEncoder] = {
    key: LabelEncoder() for key in filt_df.keys()
    if is_string_dtype(filt_df[key])
}
for key, le in le_dict.items():
    print(f"translating {key}")
    filt_df[key] = le.fit_transform(filt_df[key])


def weighted_eucl(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum((p1 - p2) ** 2))


time_start = time.time()
# res = DBSCAN(metric=weighted_eucl).fit_predict(imp_df.to_numpy())
# res = DBSCAN().fit_predict(important_cols_df.to_numpy())
filt_df.loc[:, 'cl'] = KMeans(n_clusters=3).fit_predict(filt_df.to_numpy())
print(f"clustering time = {time.time() - time_start}")

filt_df.loc[:, 'time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                        for task_time in list(filt_df['ElapsedRaw'])]

cl_stat_df = []
for cl in filt_df['cl'].unique():
    curr_cl_df = filt_df[filt_df['cl'] == cl]
    for curr_el_range in curr_cl_df['time_elapsed_range'].unique():
        curr_dumb_df = curr_cl_df[curr_cl_df['time_elapsed_range'] == curr_el_range]
        cl_stat_df.append(
            {
                'cl': cl,
                'range': curr_el_range,
                'tasks(%)': len(curr_dumb_df) / len(filt_df)
            }
        )

cl_stat_df = pd.DataFrame(cl_stat_df).sort_values(['cl', 'range'])
