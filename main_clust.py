import numpy as np
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import time
from lib.time_ranges import get_time_range_symb
from typing import Dict
from lib.kmeans_lu import KMeansLU

df = pd.read_csv('sk-full-data/fair_ds/train.csv')
y_key = 'ElapsedRaw'
x_keys = [key for key in df.keys() if key != y_key]
filt_df = df[[*x_keys, y_key]]
filt_df = filt_df.drop(columns=['MinMemoryNode']).dropna(axis=1) if 'MinMemoryNode' in filt_df.keys() else filt_df

le_dict: Dict[str, LabelEncoder] = {
    key: LabelEncoder() for key in filt_df.keys()
    if is_string_dtype(filt_df[key])
}
for key, le in le_dict.items():
    print(f"translating {key}")
    filt_df[key] = le.fit_transform(filt_df[key])

time_start = time.time()
filt_df = filt_df.iloc[:10_000]
# filt_df.loc[:, 'cl'] = KMeans(n_clusters=3, random_state=42).fit_predict(filt_df.to_numpy())
filt_df.loc[:, 'cl'] = KMeansLU(n_clusters=3, random_state=42).fit_predict(filt_df.to_numpy())
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
