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


def get_t_from_y(y) -> List[float]:
    return [y_ex[1] for y_ex in y]


def translate_censored_data(df: pd.DataFrame) -> pd.DataFrame:
    # TODO add correct processing for all events
    df.loc[:, 'event'] = 0
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 1
    df.loc[random.choice(df.index), 'event'] = 0
    return df
