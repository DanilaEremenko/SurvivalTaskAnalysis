import time

import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd

df = pd.read_csv('sk-full-data/full_data.csv')
features_keys = ['NumTasks', 'CPUs/Task', 'Priority', 'TimelimitRaw']
y_key = 'ElapsedRaw'
important_cols_df = df[[*features_keys, y_key]]


def weighted_eucl(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum((p1 - p2) ** 2))


time_start = time.time()
# res = DBSCAN(metric=weighted_eucl).fit_predict(imp_df.to_numpy())
res = DBSCAN().fit_predict(important_cols_df.to_numpy())
print(f"clustering time = {time.time() - time_start}")
