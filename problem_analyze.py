import pandas as pd

from time_ranges import get_time_range_symb

src_df = pd.read_csv('sk-full-data/fair_ds/test.csv')
src_df['TimelimitReg'] = pd.read_csv('sk-full-data/fair_ds/y_pred_reg.csv')['y_pred']
src_df['TimelimitSurv'] = pd.read_csv('sk-full-data/fair_ds/y_pred_surv.csv')['y_pred']

src_df['time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                for task_time in list(src_df['ElapsedRaw'])]

src_df['time_limit_range'] = [get_time_range_symb(task_time=task_time)
                              for task_time in list(src_df['TimelimitRaw'])]

src_df['time_limit_reg_range'] = [get_time_range_symb(task_time=task_time)
                                  for task_time in list(src_df['TimelimitReg'])]

src_df['time_limit_surv_range'] = [get_time_range_symb(task_time=task_time)
                                   for task_time in list(src_df['TimelimitSurv'])]

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

        curr_surv_df = curr_el_df[curr_el_df['time_limit_surv_range'] == pred_limit]
        curr_surv_gap = (curr_reg_df['TimelimitSurv'] - curr_reg_df['ElapsedRaw']).sum()

        confusion_df.append(
            {
                'y_true': gt_limit,
                'y_predicted': pred_limit,
                'tasks dumb (part)': len(curr_dumb_df) / len(curr_el_df),
                'gap dumb (part)': curr_dumb_gap / SUMMARY_GAP,
                'tasks reg (part)': len(curr_reg_df) / len(curr_el_df),
                'gap reg (part)': curr_reg_gap / SUMMARY_GAP,
                'tasks surv (part)': len(curr_surv_df) / len(curr_el_df),
                'gap surv (part)': curr_surv_gap / SUMMARY_GAP
            }
        )

confusion_df = pd.DataFrame(confusion_df).sort_values(['y_true', 'y_predicted'], ascending=False)
