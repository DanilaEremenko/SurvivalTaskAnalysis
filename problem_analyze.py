import pandas as pd

from drawing import draw_corr_sns
from time_ranges import get_time_range_symb

src_df = pd.read_csv('sk-full-data/fair_ds/test.csv')
src_df['TimelimitReg'] = pd.read_csv('sk-full-data/fair_ds/y_pred_reg.csv')['y_pred']
src_df['TimelimitSurv1'] = pd.read_csv('sk-full-data/fair_ds/y_pred_surv_big_trees.csv')['y_pred']
src_df['TimelimitSurv2'] = pd.read_csv('sk-full-data/fair_ds/y_pred_surv_big_trees_my_pred.csv')['y_pred']

src_df['time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                for task_time in list(src_df['ElapsedRaw'])]

src_df['time_limit_range'] = [get_time_range_symb(task_time=task_time)
                              for task_time in list(src_df['TimelimitRaw'])]

src_df['time_limit_reg_range'] = [get_time_range_symb(task_time=task_time)
                                  for task_time in list(src_df['TimelimitReg'])]

src_df['time_limit_surv_range1'] = [get_time_range_symb(task_time=task_time)
                                    for task_time in list(src_df['TimelimitSurv1'])]

src_df['time_limit_surv_range2'] = [get_time_range_symb(task_time=task_time)
                                    for task_time in list(src_df['TimelimitSurv2'])]

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
for gt_limit in src_df['time_elapsed_range'].unique():
    curr_el_df = src_df[src_df['time_elapsed_range'] == gt_limit]
    for pred_limit in src_df['time_elapsed_range'].unique():
        curr_dumb_df = curr_el_df[curr_el_df['time_limit_range'] == pred_limit]
        curr_dumb_gap = (curr_dumb_df['TimelimitRaw'] - curr_dumb_df['ElapsedRaw']).sum()

        curr_reg_df = curr_el_df[curr_el_df['time_limit_reg_range'] == pred_limit]
        curr_reg_gap = (curr_reg_df['TimelimitReg'] - curr_reg_df['ElapsedRaw']).sum()

        curr_surv1_df = curr_el_df[curr_el_df['time_limit_surv_range1'] == pred_limit]
        curr_surv1_gap = (curr_reg_df['TimelimitSurv1'] - curr_reg_df['ElapsedRaw']).sum()

        curr_surv2_df = curr_el_df[curr_el_df['time_limit_surv_range2'] == pred_limit]
        curr_surv2_gap = (curr_reg_df['TimelimitSurv2'] - curr_reg_df['ElapsedRaw']).sum()

        confusion_df.append(
            {
                'y_true': gt_limit,
                'y_predicted': pred_limit,

                'tasks dumb (part)': len(curr_dumb_df) / len(curr_el_df),
                # 'gap dumb (part)': curr_dumb_gap / SUMMARY_GAP,

                'tasks reg (part)': len(curr_reg_df) / len(curr_el_df),
                # 'gap reg (part)': curr_reg_gap / SUMMARY_GAP,

                'tasks surv 1 (part)': len(curr_surv1_df) / len(curr_el_df),
                # 'gap surv 1 (part)': curr_surv1_gap / SUMMARY_GAP,

                'tasks surv 2 (part)': len(curr_surv2_df) / len(curr_el_df),
                # 'gap surv 2 (part)': curr_surv2_gap / SUMMARY_GAP
            }
        )

confusion_df = pd.DataFrame(confusion_df).sort_values(['y_true', 'y_predicted'], ascending=False)

model_names = ['human', 'reg model', 'surv model (lib predicts)', 'surv model (manual predicts)']
model_keys = ['TimelimitRaw', 'TimelimitReg', 'TimelimitSurv1', 'TimelimitSurv2']
for model_name, model_key in zip(model_names, model_keys):
    draw_corr_sns(
        group_df=src_df,
        x_key='ElapsedRaw', y_key=model_key,
        x_title='Elapsed Time', y_title=f'Predicted Time',
        add_rmse=False, add_mae=False, add_mae_perc=False, kind='reg',
        res_dir=None, title=f'{model_name} predictions',
        add_bounds=True
    )
