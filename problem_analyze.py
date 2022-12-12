import pandas as pd

from experiments_config import EXP_PATH, CL_DIR, CL_MODE
from lib.drawing import draw_corr_sns, draw_df_as_image
from lib.time_ranges import get_time_range_symb, TIME_RANGES
import matplotlib

font = {
    'family': 'Times New Roman',
    'size': 14
}
matplotlib.rc('font', **font)

compared_models = [
    {
        'model_name': 'user',
        'pred_path': None,
        'src_key': 'TimelimitRaw'
    },
    {
        'model_name': 'reg rf',
        'pred_path': f'{EXP_PATH}/y_pred_reg_rf.csv',

    },
    {
        'model_name': 'reg lgbm',
        'pred_path': f'{EXP_PATH}/y_pred_reg_lgbm.csv'
    },
    {
        'model_name': 'surv rf',
        'pred_path': f'{EXP_PATH}/y_pred_surv.csv'
    },
    {
        'model_name': 'cl+reg rf',
        'pred_path': f'{EXP_PATH}/y_pred_cl_{CL_MODE}_{CL_DIR}.csv'
    }
]

src_df = pd.read_csv(f'{EXP_PATH}/test.csv')

for compared_dict in compared_models:
    if compared_dict['pred_path'] is not None:
        src_df[compared_dict['model_name']] = pd.read_csv(compared_dict['pred_path'])['y_pred']
    else:
        src_df[compared_dict['model_name']] = src_df[compared_dict['src_key']] * 60

src_df['time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                for task_time in list(src_df['ElapsedRaw'])]

for compared_dict in compared_models:
    src_df[compared_dict['model_name'] + ' range'] = [get_time_range_symb(task_time=task_time)
                                                      for task_time in list(src_df[compared_dict['model_name']])]

time_elapsed_agg_df = pd.DataFrame([
    {
        'range': tr,
        'percent_tasks': len(src_df[src_df['time_elapsed_range'] == tr]) / len(src_df),
        'percent_time': src_df[src_df['time_elapsed_range'] == tr]['ElapsedRaw'].sum() / src_df['ElapsedRaw'].sum()
    }
    for tr in src_df['time_elapsed_range'].unique()]
).sort_values('percent_time', ascending=False)

SUMMARY_GAP = (src_df['TimelimitRaw'] - src_df['ElapsedRaw']).sum()
SUMMARY_ELAPSED = src_df['ElapsedRaw'].sum()

confusion_df = []

confusion_matrixes = {model_dict['model_name']: pd.DataFrame(index=TIME_RANGES.keys(), columns=TIME_RANGES.keys())
                      for model_dict in compared_models}
for gt_limit in src_df['time_elapsed_range'].unique():
    curr_el_df = src_df[src_df['time_elapsed_range'] == gt_limit]
    for pred_limit in src_df['time_elapsed_range'].unique():
        curr_dict = {
            'y_true': gt_limit,
            'y_predicted': pred_limit,
        }
        for model_dict in compared_models:
            curr_df = curr_el_df[curr_el_df[model_dict['model_name'] + ' range'] == pred_limit]
            curr_gap = (curr_df[model_dict['model_name']] - curr_df['ElapsedRaw']).sum()
            curr_percent_prec = round(len(curr_df) / len(curr_el_df), 4)
            curr_dict[f"{model_dict['model_name']} tasks (part)"] = curr_percent_prec
            confusion_matrixes[model_dict['model_name']].loc[gt_limit, pred_limit] = curr_percent_prec
            # curr_dict[f"{model_dict['model_name']} gap (part)"] = curr_gap / SUMMARY_GAP

        confusion_df.append(curr_dict)

confusion_matrixes = {key: val.astype(float) for key, val in confusion_matrixes.items()}
confusion_df = pd.DataFrame(confusion_df).sort_values(['y_true', 'y_predicted'], ascending=False)

for model_key, matrix_df in confusion_matrixes.items():
    draw_corr_sns(
        group_df=src_df,
        y_key='ElapsedRaw', x_key=model_key,
        y_title='Elapsed Time', x_title='Predicted Time',
        add_rmse=False, add_mae=False, add_mae_perc=False, kind='reg',
        res_dir=None, title=f'{model_key} predictions',
        add_bounds=True
    )
    draw_df_as_image(df=matrix_df, title=model_key)
