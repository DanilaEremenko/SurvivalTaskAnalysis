from typing import Dict

import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest

x_all, y_all = load_gbsg2()

grade_str = x_all.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

x_all = x_all.drop("tgrade", axis=1)

le_dict: Dict[str, LabelEncoder] = {
    key: LabelEncoder() for key in x_all.keys()
    if is_string_dtype(x_all[key]) or is_categorical_dtype(x_all[key])

}

for key, le in le_dict.items():
    x_all[key] = le.fit_transform(x_all[key])

rsf = RandomSurvivalForest(
    n_estimators=10,
    min_samples_split=10,
    min_samples_leaf=15,
    n_jobs=4,
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

rsf.fit(x_train, y_train)

rsf.score(x_test, y_test)
