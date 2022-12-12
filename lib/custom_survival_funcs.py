import random
from typing import List

import numpy as np
import pandas as pd
from numba import njit
from sksurv.ensemble import RandomSurvivalForest


@njit
def get_event_time_manual(event_times: np.ndarray, probs: np.ndarray) -> List[float]:
    res_arr = []
    for i, curr_probs in enumerate(probs):
        min_prob = curr_probs.min()
        for et, prob in zip(event_times, curr_probs):
            if prob == min_prob:
                res_arr.append(et)
                break
        # if len(res_arr) != i + 1:
        #     raise Exception('No prob < 0.1 in prob vector')
    return res_arr


def batch_surv_time_pred(model: RandomSurvivalForest, X: pd.DataFrame, batch_size=5000) -> np.ndarray:
    return np.concatenate([
        get_event_time_manual(
            event_times=model.event_times_,
            probs=model.predict_survival_function(X[start:start + batch_size], return_array=True)
        )
        for start in range(0, len(X), 5000)
    ])


def batch_risk_score_pred(model: RandomSurvivalForest, X: pd.DataFrame, batch_size=5000) -> np.ndarray:
    return np.concatenate([
        model.predict(X[start:start + batch_size])
        for start in range(0, len(X), batch_size)
    ])


def get_t_from_y(y) -> np.ndarray:
    return np.array([y_ex[1] for y_ex in y])


def add_events_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, 'event'] = -1
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'RUNNING', 'event'] = 0
    df.loc[df['State'] == 'FAILED', 'event'] = 0
    df.loc[df['State'] == 'CANCELLED', 'event'] = 0
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 0
    df.loc[df['State'] == 'NODE_FAIL', 'event'] = 0
    df.loc[df['State'] == 'OUT_OF_MEMORY', 'event'] = 0

    for state in df['State'].unique():
        if 'CANCELLED by' in state:
            df.loc[df['State'] == state, 'event'] = 0

    assert all([event != -1 for event in df['event']])
    df.loc[random.choice(df.index), 'event'] = 0
    return df
