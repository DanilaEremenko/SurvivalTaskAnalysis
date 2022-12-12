from collections import Counter

import numpy as np
import warnings

# TODO I'm sorry but I'll remove it later

from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time

from experiments_config import EXP_PATH, CL_MODE, CL_DIR
from lib.time_ranges import get_time_range_symb
from typing import Dict, List, Optional
from lib.kmeans_lu import KMeansLU
import matplotlib.pyplot as plt

CL_RES_DIR = Path(f'{EXP_PATH}/clustering_{CL_MODE}_{CL_DIR}')
CL_RES_DIR.mkdir(exist_ok=True)

WITH_NORMALIZATION = False


def get_stat_df_by_key(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    df.loc[:, 'time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                       for task_time in list(df['ElapsedRaw'])]

    stat_df = []
    for cl in df[group_key].unique():
        curr_cl_df = df[df[group_key] == cl]
        for curr_el_range in curr_cl_df['time_elapsed_range'].unique():
            curr_dumb_df = curr_cl_df[curr_cl_df['time_elapsed_range'] == curr_el_range]
            stat_df.append(
                {
                    group_key: cl,
                    'range': curr_el_range,
                    'tasks(%)': len(curr_dumb_df) / len(df)
                }
            )
    df.drop(columns='time_elapsed_range', inplace=True)
    return pd.DataFrame(stat_df).sort_values([group_key, 'range'])


def cluster_df(df: pd.DataFrame, n_clusters: int):
    time_start = time.time()
    if CL_MODE == 'km':
        k = KMeans(n_clusters=n_clusters, random_state=42).fit(df[CLUST_KEYS].to_numpy())
    elif CL_MODE == 'kmlu':
        k = KMeansLU(n_clusters=n_clusters, random_state=42).fit(df.to_numpy())
    else:
        raise Exception(f"Unexpected clustering mode = {CL_MODE}")

    print(f"clustering time = {time.time() - time_start:.2f}")
    return k


def normalize_df(df: pd.DataFrame, norm_dict: Dict[str, LabelEncoder]):
    if WITH_NORMALIZATION:
        for key, n in norm_dict.items():
            df.loc[:, key] = n.transform(np.array(df[key]).reshape(-1, 1))


def inverse_df(df: pd.DataFrame, norm_dict: Dict[str, LabelEncoder]):
    if WITH_NORMALIZATION:
        for key, n in norm_dict.items():
            df.loc[:, key] = n.inverse_transform(np.array(df[key]).reshape(-1, 1))
            df.loc[:, key] = df[key].astype(dtype=int)


def print_one_cl_dist(df: pd.DataFrame):
    cl_dist = dict(Counter([get_time_range_symb(task_time=task_time)
                            for task_time in list(df['ElapsedRaw'])]))
    cl_dist_df = pd.DataFrame({'time_range': cl_dist.keys(), 'tasks(%)': cl_dist.values()}) \
        .sort_values('time_range', ascending=True)
    cl_dist_df['tasks(%)'] /= len(df)
    print(f'classes distribution in cluster: \n{cl_dist_df.round(2)}')


def print_tasks_dist(df: pd.DataFrame, cl_key: Optional[str] = None):
    if cl_key is not None:
        print(f"----------- tasks distribution by {cl_key} -----------------------")
        for i, cl_name in enumerate(df[cl_key].unique()):
            curr_cl_df = df[df[cl_key] == cl_name]
            print(f'cl size = {len(curr_cl_df)} ({100 * len(curr_cl_df) / len(df):.2f}%)')
            print_one_cl_dist(df=curr_cl_df)
            print()
    else:
        print(f"----------- all tasks distribution by {cl_key} -------------------")
        print_one_cl_dist(df=df)
        print()


def rec_cluster(df: pd.DataFrame, centroids_dict: Dict[int, List], norm_dict: Dict[str, LabelEncoder]):
    normalize_df(df, norm_dict)

    clust_levels = [int(key.split('_')[1][1:]) for key in df.keys() if 'cl_l' in key]
    print(f"------------------------- clustering step {len(clust_levels) + 1} ------------------------")

    if len(clust_levels) == 0:  # first clustering
        next_cl_key = 'cl_l1'
        k = cluster_df(df=df, n_clusters=2)
        df.loc[:, next_cl_key] = k.labels_
        for i in range(2):
            centroids_dict[i] = k.cluster_centers_[i]
        centroids_df = centroids_dict_to_df(centroids_dict)
        inverse_df(centroids_df, norm_dict)
        centroids_df.to_csv(f'{CL_RES_DIR}/train_centroids_l1.csv')

    else:  # hierarchical clustering of current biggest cluster
        last_cl_lvl = max(clust_levels)

        last_cl_key = f'cl_l{last_cl_lvl}'
        next_cl_key = f'cl_l{last_cl_lvl + 1}'

        df.loc[:, next_cl_key] = df[last_cl_key]
        cl_dist_df = pd.DataFrame(
            [{'cl_val': cl_val, 'cl_part': len(df[df[last_cl_key] == cl_val]) / len(df)}
             for cl_val in df[last_cl_key].unique()]
        ).sort_values('cl_part', ascending=False).reset_index()
        cl_val_shift = cl_dist_df['cl_val'].max() + 1
        biggest_clust_key, biggest_clust_part = cl_dist_df.iloc[0][['cl_val', 'cl_part']]
        smallest_clust_key, smallest_clust_part = cl_dist_df.iloc[-1][['cl_val', 'cl_part']]

        print('splitting biggest clust')
        print(f'biggest clust key  = {biggest_clust_key}')
        print(f'biggest clust part = {biggest_clust_part:.2f}')
        print(f'smallest clust key  = {smallest_clust_key}')
        print(f'smallest clust part = {smallest_clust_part:.2f}')

        k = cluster_df(df=df[df[last_cl_key] == biggest_clust_key], n_clusters=2)
        df.loc[df[df[last_cl_key] == biggest_clust_key].index, next_cl_key] = k.labels_ + cl_val_shift
        del centroids_dict[biggest_clust_key]
        for i in range(2):
            centroids_dict[i + cl_val_shift] = k.cluster_centers_[i]
        centroids_df = centroids_dict_to_df(centroids_dict)
        inverse_df(centroids_df, norm_dict)
        centroids_df.to_csv(f'{CL_RES_DIR}/train_centroids_l{last_cl_lvl + 1}.csv')

    inverse_df(df, norm_dict)

    print_tasks_dist(df=df, cl_key=next_cl_key)


def flat_cluster(df: pd.DataFrame, norm_dict: Dict[str, LabelEncoder]):
    normalize_df(df, norm_dict)
    cl_pref = 'cl_l'
    clust_levels = [int(key.split('_')[1][1:]) for key in df.keys() if 'cl_l' in key]
    print(f"------------------------- clustering step {len(clust_levels) + 1} ------------------------")
    next_cl_key = f'{cl_pref}{len(clust_levels) + 1}'

    n_clusters = len(clust_levels) + 2
    k = cluster_df(df=df, n_clusters=n_clusters)
    df.loc[:, next_cl_key] = k.labels_
    centroids_df = centroids_dict_to_df(centroids_dict={i: k.cluster_centers_[i] for i in range(n_clusters)})
    inverse_df(centroids_df, norm_dict)
    centroids_df.to_csv(f'{CL_RES_DIR}/train_centroids_l{len(clust_levels) + 1}.csv')

    inverse_df(df, norm_dict)
    print_tasks_dist(df=df, cl_key=next_cl_key)


def centroids_dict_to_df(centroids_dict: Dict[int, List]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'cl': cl_id,
            **{
                feature_key: centroid_coord
                for feature_key, centroid_coord in zip(CLUST_KEYS, cl_coords)
            }
        }
        for cl_id, cl_coords in centroids_dict.items()
    )


def draw_tsne_matplot(
        features_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        save_pref: Optional[str] = None
):
    if features is not None:
        tsne_arr = TSNE(perplexity=2 if len(features_df) < 30 else 30).fit_transform(features_df[features])
    else:
        tsne_arr = TSNE(perplexity=2 if len(features_df) < 30 else 30).fit_transform(features_df)

    plt.rcParams["figure.figsize"] = (10, 10)

    features_df = features_df.reset_index(inplace=False)
    for cl_name in list(features_df['cl'].unique()):
        curr_class_df = features_df[features_df['cl'] == cl_name]
        # class_centroid = curr_class_df[features].mean().reset_index()[0]
        # class_distances = [np.linalg.norm(class_centroid - np.array(tup)) for tup in
        #                    curr_class_df[features].itertuples(index=False)]
        plt.scatter(
            tsne_arr[curr_class_df.index, 0],
            tsne_arr[curr_class_df.index, 1],
            label=f"{cl_name}",
            edgecolor="black",
            s=200
        )
    plt.legend(loc="upper right")

    if save_pref is None:
        plt.show()
    else:
        plt.savefig(f"{save_pref}.png", dpi=100)
        plt.clf()


if __name__ == '__main__':
    # -------------------------------------------------------------
    # ---------------------- data parsing -------------------------
    # -------------------------------------------------------------
    df = pd.read_csv(f'{EXP_PATH}/train.csv', index_col=0)
    y_key = 'ElapsedRaw'
    x_keys = [key for key in df.keys() if key not in [y_key, 'State']]
    filt_df = df[[*x_keys, y_key]]

    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }
    for key, le in le_dict.items():
        print(f"translating {key}")
        filt_df[key] = le.fit_transform(filt_df[key])

    CLUST_KEYS = filt_df.keys()

    norm_dict: Dict[str, LabelEncoder] = {
        key: MinMaxScaler().fit(np.array(filt_df[key]).reshape(-1, 1))
        for key in CLUST_KEYS
        # if 'cl_l' not in key
        # if is_numeric_dtype(filt_df[key])
    }
    # -------------------------------------------------------------
    # ----------------- clustering & analyze ----------------------
    # -------------------------------------------------------------
    # filt_df = filt_df.iloc[:10_000]
    centroids_dict = {}
    print_tasks_dist(df=filt_df)
    n_splits = 4
    for i in range(n_splits):
        if CL_DIR == 'recurs':
            rec_cluster(filt_df, centroids_dict, norm_dict)
        elif CL_DIR == 'flat':
            flat_cluster(filt_df, norm_dict)
        else:
            raise Exception(f"Unexpected clust direction = {CL_DIR}")

    # filt_df['cl_l0'] = 0
    # stat_df = get_stat_df_by_key(filt_df, group_key='cl_l4')

    # draw_tsne_matplot(features_df=centroids_dict_to_df(centroids_dict))

    # check clust dist
    # sorted([round(filt_df[filt_df['cl_l4'] == cl]['ElapsedRaw'].mean() / 3600, 2) for cl in filt_df['cl_l4'].unique()])
    # [round(filt_df[filt_df['cl_l4'] == cl]['ElapsedRaw'].mean() / 3600, 2) for cl in filt_df['cl_l4'].unique()]
    # [round(len(filt_df[filt_df['cl_l4'] == cl]) / len(filt_df), 2) for cl in filt_df['cl_l4'].unique()]
    # centroids_dict_to_df(centroids_dict)[['cl', 'ElapsedRaw']]
    # list(round(centroids_dict_to_df(centroids_dict)['ElapsedRaw'] / 3600, 2))

    res_df = pd.merge(left=df, right=filt_df[[key for key in filt_df if 'cl_l' in key]],
                      left_index=True, right_index=True)
    res_df.to_csv(f'{CL_RES_DIR}/train_clustered.csv')
