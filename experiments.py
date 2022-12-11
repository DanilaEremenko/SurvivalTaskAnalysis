from datetime import datetime
from pathlib import Path

from lib.spliters import DatesSplit

# DATA_SPLITER = DatesSplit(
#     name='train_all_test_nov',
#     train_left=datetime(year=2022, month=6, day=1),
#     train_right=datetime(year=2022, month=10, day=17),
#     test_left=datetime(year=2022, month=10, day=17),
#     test_right=datetime(year=2022, month=11, day=18)
#
# )

DATA_SPLITER = DatesSplit(
    name='train_autumn_test_nov',
    train_left=datetime(year=2022, month=9, day=1),
    train_right=datetime(year=2022, month=10, day=17),
    test_left=datetime(year=2022, month=10, day=17),
    test_right=datetime(year=2022, month=11, day=18)

)

# TASKS_GROUP = 'geov'
TASKS_GROUP = 'nogeov'

MODELS_MODE = 'search'
# MODELS_MODE = 'predict'


def get_ds_path_name() -> str:
    return f'ds_group={TASKS_GROUP}, split={DATA_SPLITER.name}'


EXP_PATH = Path(f'sk-full-data/{get_ds_path_name()}')
EXP_PATH.mkdir(exist_ok=True, parents=True)

# CL_MODE = 'km'
CL_MODE = 'kmlu'

# CL_DIR = 'recurs'
CL_DIR = 'flat'
