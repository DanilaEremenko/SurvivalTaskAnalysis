import random
from typing import List

import numpy as np
import pandas as pd


def get_event_time_manual(event_times: np.ndarray, probs: np.ndarray) -> List[float]:
    res_arr = []
    for i, curr_probs in enumerate(probs):
        for et, prob in zip(event_times, curr_probs):
            if prob == curr_probs.min():
                res_arr.append(et)
                break
        # if len(res_arr) != i + 1:
        #     raise Exception('No prob < 0.1 in prob vector')
    return res_arr


def translate_censored_data(df: pd.DataFrame) -> pd.DataFrame:
    # TODO add correct processing for all events
    df.loc[:, 'event'] = 0
    df.loc[df['State'] == 'COMPLETED', 'event'] = 1
    df.loc[df['State'] == 'TIMEOUT', 'event'] = 1
    df.loc[random.choice(df.index), 'event'] = 0
    return df
