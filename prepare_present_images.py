from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from pandas.core.dtypes.common import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from experiments_config import EXP_PATH
from lib.custom_survival_funcs import add_events_to_df, get_event_time_manual
from lib.drawing import get_random_color

font = {'family': 'Times New Roman',
        'size': 32}
matplotlib.rc('font', **font)


class ExpSurvDesc:
    def __init__(
            self,
            res_dir: str,
            train_file: str,
            test_file: str,
            y_key: str,
            event_key: str,
            translate_func
    ):
        self.res_dir = Path(res_dir)
        self.train_file = train_file
        self.test_file = test_file
        self.y_key = y_key
        self.event_key = event_key
        self.translate_func = translate_func


if __name__ == '__main__':
    ################################################
    # ------------ exp descriptions  ---------------
    ################################################
    exp_desc = ExpSurvDesc(
        res_dir=f"{EXP_PATH}/psearch_surv_elapsed_time",
        train_file=f'{EXP_PATH}/train.csv',
        test_file=f'{EXP_PATH}/test.csv',
        y_key='ElapsedRaw',
        event_key='event',
        translate_func=add_events_to_df
    )

    exp_desc.res_dir.mkdir(exist_ok=True)

    ####################################################################################################################
    # ----------------------------------------- data_type processing  --------------------------------------------------
    ####################################################################################################################
    train_df = pd.read_csv(exp_desc.train_file, index_col=0)
    test_df = pd.read_csv(exp_desc.test_file, index_col=0)

    # State is neccessary for event formation
    train_df = exp_desc.translate_func(train_df)
    test_df = exp_desc.translate_func(test_df)

    train_df.drop(columns=['State'], inplace=True)
    test_df.drop(columns=['State'], inplace=True)

    # initialize label encoders
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in train_df.keys()
        if is_string_dtype(train_df[key])
    }

    x_train = train_df.drop(columns=[exp_desc.y_key, exp_desc.event_key])
    y_train_src = train_df[[exp_desc.y_key, exp_desc.event_key]]

    x_test = test_df.drop(columns=[exp_desc.y_key, exp_desc.event_key])
    y_test_src = test_df[[exp_desc.y_key, exp_desc.event_key]]

    for key, le in le_dict.items():
        x_train[key] = le.fit_transform(x_train[key])
        x_test[key] = le.fit_transform(x_test[key])

    # x_all = x_all.iloc[:1_000]
    # y_all_tr = y_all_tr[:1_000]

    y_train = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_train_src)
    y_test = Surv.from_dataframe(event='event', time='ElapsedRaw', data=y_test_src)

    ####################################################################################################################
    # -------------------------------------- deep survival analyze -----------------------------------------------------
    ####################################################################################################################
    rsf = RandomSurvivalForest(
        n_estimators=150,
        bootstrap=True,
        max_samples=500,
        min_samples_leaf=4,
        max_depth=10,
        n_jobs=4,
        random_state=42
    )
    rsf.fit(X=x_train, y=y_train)
    x_test = x_train
    y_test_src = y_train_src
    y_test_src_df_sorted = y_test_src.sort_values('ElapsedRaw')
    event = 1
    y_test_src_df_sel = pd.concat([
        y_test_src_df_sorted[
            (y_test_src_df_sorted['ElapsedRaw'] > 10 * 60) & (y_test_src_df_sorted['ElapsedRaw'] < 20 * 60)
            & (y_test_src_df_sorted['event'] == event)].iloc[[0, 5, 10]],

        y_test_src_df_sorted[
            (y_test_src_df_sorted['ElapsedRaw'] > 9 * 3600) & (y_test_src_df_sorted['ElapsedRaw'] < 10 * 3600)
            & (y_test_src_df_sorted['event'] == event)].iloc[[0, 5, 10]],

        y_test_src_df_sorted[
            (y_test_src_df_sorted['ElapsedRaw'] > 2 * 24 * 3600) & (y_test_src_df_sorted['ElapsedRaw'] < 3 * 24 * 3600)
            & (y_test_src_df_sorted['event'] == event)].iloc[[0, 5, 10]]
    ]
    )

    x_test_sel = x_test.loc[y_test_src_df_sel.index]
    y_pred = rsf.predict_survival_function(x_test_sel, return_array=True)
    ####################################################################################################################
    # -------------------------------------------------- drawing gif ---------------------------------------------------
    ####################################################################################################################
    fig = plt.figure(figsize=(4 * 5, 3 * 5))
    ax = plt.axes(xlim=(-10, 200), ylim=(-0.1, 1))
    data_type = 'censored' if event == 0 else 'not censored'
    # ax.set_title(f'{data_type} data rsf survival functions')
    ax.set_title(f'Функции выживаемости')
    ax.set_xlabel('Время решения (часы)')
    ax.set_ylabel('Вероятность завершения')
    # ax.grid(True)
    # ax.legend(handles=[f"Elapsed time = {task / 3600:.2f}" for task in list(y_test_src_df_sel['ElapsedRaw'])])
    lines = [ax.plot([], [])[0] for i in range(len(y_pred))]
    GROUP_COLORS = ["#53EF70", "#FFD457", "#FF4848"]
    COLORS = [group_color for group_color in GROUP_COLORS for repeat in range(3)]
    assert len(COLORS) == len(y_test_src_df_sel)
    # COLORS = [get_random_color() for i in range(len(y_test_src_df_sel))]

    ####################################################################################################################
    # -------------------------------------------------- drawing gif ---------------------------------------------------
    ####################################################################################################################
    # def init():
    #     for line in lines:
    #         line.set_data([], [])
    #     return lines
    #
    #
    # def animate(i):
    #     step_lines = []
    #     for pred_i, (line, curr_pred, color) in enumerate(zip(lines, y_pred, COLORS)):
    #         x = rsf.event_times_[:i * step] / 3600
    #         y = curr_pred[:i * step]
    #         label = f"Фактическое время решения = {y_test_src_df_sel.iloc[pred_i]['ElapsedRaw'] / 3600:.2f} часов"
    #         step_lines.append(plt.step(x, y, where="post", label=label, linewidth=4, color=color)[0])
    #         line.set_data(x, y)
    #         # line.set_label(f'Tasks time')
    #     ax.legend(handles=step_lines, loc='upper right')
    #     fig.tight_layout()
    #     return step_lines
    #
    #
    # step = 40
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                frames=len(rsf.event_times_) // step, interval=20, blit=True)
    # f = f'rsf {data_type}.gif'
    # writergif = animation.PillowWriter(fps=5)
    # anim.save(f, writer=writergif)
    ####################################################################################################################
    # -------------------------------------------------- drawing last frame --------------------------------------------
    ####################################################################################################################
    for pred_i, (line, curr_pred, color) in enumerate(zip(lines, y_pred, COLORS)):
        x = rsf.event_times_ / 3600
        y = curr_pred
        y_true = y_test_src_df_sel.iloc[pred_i]['ElapsedRaw'] / 3600
        y_math_exp = (y * rsf.event_times_).mean() / 3600
        y_min_prob = get_event_time_manual(event_times=rsf.event_times_, probs=np.expand_dims(y, 0))[0] / 3600
        label = f"Фактическое время решения = {y_true:.2f} часов"
        plt.step(x, y, where="post", label=label, linewidth=4, color=color)[0]
        plt.plot([y_true, y_true], [0, 1], '--', linewidth=2, color=color)
    plt.legend(loc='upper right')
    plt.show()
