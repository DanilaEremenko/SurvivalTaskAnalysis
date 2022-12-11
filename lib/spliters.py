from datetime import datetime
from typing import Tuple

import pandas as pd


class TrainTestSplit:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        raise Exception("Should be implemented")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise Exception("Should be implemented")


class DatesSplit(TrainTestSplit):
    def __init__(
            self,
            name: str,
            train_left: datetime,
            train_right: datetime,
            test_left: datetime,
            test_right: datetime
    ):
        super().__init__()
        self._name = name
        self.train_left = train_left
        self.train_right = train_right
        self.test_left = test_left
        self.test_right = test_right

    @property
    def name(self) -> str:
        return self._name

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df['SubmitDatetime'] = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in df['SubmitTime']]
        train_df = df[(df['SubmitDatetime'] >= self.train_left) & (df['SubmitDatetime'] < self.train_right)]
        test_df = df[(df['SubmitDatetime'] >= self.test_left) & (df['SubmitDatetime'] < self.test_right)]

        for df in [df, train_df, test_df]:
            df.drop(columns='SubmitDatetime', inplace=True)

        return train_df, test_df
