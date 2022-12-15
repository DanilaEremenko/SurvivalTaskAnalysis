import random
from typing import List

import numpy as np
import pandas as pd
from numba import njit
from sksurv.ensemble import RandomSurvivalForest


@njit
def get_event_time_last_right(event_times: np.ndarray, probs: np.ndarray) -> float:
    min_prob = probs.min()
    for et, prob in zip(event_times, probs):
        if prob == min_prob:
            return et


@njit
def get_event_time_math_exp(event_times: np.ndarray, probs: np.ndarray) -> float:
    return (event_times * probs).mean()


def batch_surv_time_pred(model: RandomSurvivalForest, X: pd.DataFrame, mode: str, batch_size=5000) -> np.ndarray:
    if mode == 'last_right':
        get_event_time_func = get_event_time_last_right
    elif mode == 'math_exp':
        get_event_time_func = get_event_time_math_exp
    else:
        raise Exception(f'Unexpected mode = {mode}')

    return np.array([
        get_event_time_func(
            event_times=model.event_times_,
            probs=probs
        )
        for start in range(0, len(X), 5000)
        for probs in model.predict_survival_function(X[start:start + batch_size], return_array=True)
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
