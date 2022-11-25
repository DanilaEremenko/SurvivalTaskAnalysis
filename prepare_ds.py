from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

src_df = pd.read_csv('sk-full-data/full_data.csv')
res_dir = Path('sk-full-data/fair_ds')
res_dir.mkdir(exist_ok=True)

ignored_keys = [
    'UserID', 'GroupID', 'JobID',
    'CPUTimeRAW', 'ElapsedRaw_mean', 'ElapsedRaw_std', 'ElapsedRawClass',
    'SizeClass',
    'AveCPU', 'AveCPU_mean', 'AveCPU_std',
    'Unnamed: 0', 'State', 'MinMemoryNode'
]

filt_df = src_df.dropna(axis=1)

filt_ignored_keys = [key for key in ignored_keys if key in filt_df.keys()]
filt_df = filt_df.drop(columns=filt_ignored_keys)

train_df, test_df = train_test_split(filt_df, test_size=0.33, random_state=42)

train_df.to_csv(f'{res_dir}/train.csv')
test_df.to_csv(f'{res_dir}/test.csv')
