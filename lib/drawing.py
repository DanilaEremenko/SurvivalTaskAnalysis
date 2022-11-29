from pathlib import Path
from typing import Optional

import plotly.express as px
import pandas as pd
import seaborn as sns
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt

from lib.losses import Losses


def get_random_color() -> str:
    return "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])


def draw_group_bars_and_boxes(df: pd.DataFrame, group_key: str, y_key: str, res_path: str):
    random.seed(42)
    row_num = 2
    fig = make_subplots(
        rows=row_num, cols=1,
        vertical_spacing=0.5 / row_num
    )
    for group_val in df[group_key].unique():
        curr_df = df[df[group_key] == group_val]
        curr_color = get_random_color()
        fig.add_trace(
            go.Bar(
                name=group_val,
                legendgroup=group_val,
                x=[group_val],
                y=[len(curr_df)],
                marker_color=curr_color
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(
                name=group_val,
                legendgroup=group_val,
                y=curr_df[y_key],
                marker_color=curr_color
            ),
            row=2, col=1
        )

    fig.update_xaxes(tickangle=30)
    fig.update_layout(title_text=f'mae percentage error grouped by {group_key}')

    fig.write_html(res_path)
    # fig.show()


def draw_corr_sns(group_df: pd.DataFrame, x_key: str, y_key: str, x_title: str, y_title: str,
                  add_rmse: bool, add_mae: bool, add_mae_perc: bool, kind: str, res_dir: Optional[Path], title: str,
                  add_bounds=False):
    if res_dir is not None:
        res_dir.mkdir(exist_ok=True, parents=True)
    print(f"drawing jointplot {res_dir}")
    jp = sns.jointplot(data=group_df, x=x_key, y=y_key, kind=kind)
    jp.set_axis_labels(x_title, y_title, fontsize=16)

    corr = Losses.r(pred=group_df[x_key], y=group_df[y_key])
    rmse = Losses.rmse(group_df[x_key], group_df[y_key])
    mae = Losses.mae(group_df[x_key], group_df[y_key])
    mae_perc = ((group_df[x_key] - group_df[y_key]).abs() / group_df[x_key]).mean() * 100

    if add_bounds:
        jp.ax_marg_y.set_ylim(0, 1e6)
        jp.ax_marg_x.set_xlim(0, 1e6)
    jp.fig.suptitle(
        f"{title}\ncorr = %.2f" % corr
        + (", rmse = %.2f" % rmse if add_rmse else '')
        + (", mae = %.2f" % mae if add_mae else '')
        + (", mae perc = %.2f" % mae_perc if add_mae_perc else '')

    )
    plt.tight_layout()
    if res_dir is not None:
        plt.savefig(f"{res_dir}/corr.png")
        plt.clf()
    else:
        plt.show()


def draw_pie_chart(df: pd.DataFrame, sum_key: str, group_key: str, mode: str, res_path: str):
    labels = list(sorted(df[group_key].unique()))
    values = [df[df[group_key] == group_val][sum_key].sum() if mode == 'sum'
              else len(df[df[group_key] == group_val])
              for group_val in labels]
    print(labels)
    print(values)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, sort=False)])
    fig.update_layout(title_text=f'Pie chart with {mode} agg mode')
    fig.write_html(res_path)
