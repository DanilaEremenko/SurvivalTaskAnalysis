import numpy as np
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import time
from lib.time_ranges import get_time_range_symb
from typing import Dict, List, Optional
from lib.kmeans_lu import KMeansLU
import matplotlib.pyplot as plt


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
    k = KMeans(n_clusters=n_clusters, random_state=42).fit(df[CLUST_KEYS].to_numpy())
    # k = KMeansLU(n_clusters=n_clusters, random_state=42).fit(df.to_numpy())
    print(f"clustering time = {time.time() - time_start}")
    return k


def rec_cluster(df: pd.DataFrame, centroids_dict: Dict[int, List]):
    clust_levels = [int(key.split('_')[1][1:]) for key in df.keys() if 'cl_l' in key]

    if len(clust_levels) == 0:
        k = cluster_df(df=df, n_clusters=2)
        df.loc[:, 'cl_l1'] = k.labels_
        for i in range(2):
            centroids_dict[i] = k.cluster_centers_[i]
    else:
        last_cl_lvl = max(clust_levels)

        last_cl_key = f'cl_l{last_cl_lvl}'
        next_cl_key = f'cl_l{last_cl_lvl + 1}'

        df.loc[:, next_cl_key] = df[last_cl_key]
        cl_dist_df = pd.DataFrame(
            [{'cl_val': cl_val, 'cl_part': len(df[df[last_cl_key] == cl_val]) / len(df)}
             for cl_val in df[last_cl_key].unique()]
        ).sort_values('cl_part', ascending=False).reset_index()
        cl_val_shift = cl_dist_df['cl_val'].max() + 1
        biggest_clust_val, biggest_clust_part = cl_dist_df.iloc[0][['cl_val', 'cl_part']]

        print('splitting')
        print(f'biggest clust key  = {biggest_clust_val}')
        print(f'biggest clust part = {round(biggest_clust_part, 2)}')

        k = cluster_df(df=df[df[last_cl_key] == biggest_clust_val], n_clusters=2)
        df.loc[df[df[last_cl_key] == biggest_clust_val].index, next_cl_key] = k.labels_ + cl_val_shift
        del centroids_dict[biggest_clust_val]
        for i in range(2):
            centroids_dict[i + cl_val_shift] = k.cluster_centers_[i]


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
    df = pd.read_csv('sk-full-data/fair_ds/train.csv')
    y_key = 'ElapsedRaw'
    x_keys = [key for key in df.keys() if key != y_key]
    filt_df = df[[*x_keys, y_key]]

    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in filt_df.keys()
        if is_string_dtype(filt_df[key])
    }
    for key, le in le_dict.items():
        print(f"translating {key}")
        filt_df[key] = le.fit_transform(filt_df[key])

    CLUST_KEYS = filt_df.keys()
    # -------------------------------------------------------------
    # ----------------- clustering & analyze ----------------------
    # -------------------------------------------------------------
    # filt_df = filt_df.iloc[:10_000]
    centroids_dict = {}
    rec_cluster(filt_df, centroids_dict)
    rec_cluster(filt_df, centroids_dict)
    rec_cluster(filt_df, centroids_dict)
    rec_cluster(filt_df, centroids_dict)

    stat_df = get_stat_df_by_key(filt_df, group_key='cl_l4')

    draw_tsne_matplot(features_df=centroids_dict_to_df(centroids_dict))
