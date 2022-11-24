import pandas as pd

from time_ranges import get_time_range_symb

src_df = pd.read_csv('sk-full-data/full_data.csv')
src_df['time_elapsed_range'] = [get_time_range_symb(task_time=task_time)
                                for task_time in list(src_df['ElapsedRaw'])]

src_df['time_limit_range'] = [get_time_range_symb(task_time=task_time)
                              for task_time in list(src_df['TimelimitRaw'])]

time_elapsed_agg_df = pd.DataFrame([
    {
        'range': tr,
        'percent_tasks': len(src_df[src_df['time_elapsed_range'] == tr]) / len(src_df),
        'percent_time': src_df[src_df['time_elapsed_range'] == tr]['ElapsedRaw'].sum() / src_df['ElapsedRaw'].sum()
    }
    for tr in src_df['time_elapsed_range'].unique()]
).sort_values('percent_time', ascending=False)

time_limit_agg_df = []
for tr_elapsed in src_df['time_elapsed_range'].unique():
    curr_el_df = src_df[src_df['time_elapsed_range'] == tr_elapsed]
    for tr_limit in src_df['time_limit_range'].unique():
        curr_el_limit_df = curr_el_df[curr_el_df['time_limit_range'] == tr_limit]
        time_limit_agg_df.append(
            {
                'y_true': tr_elapsed,
                'y_predicted': tr_limit,
                'percent_tasks': len(curr_el_limit_df) / len(curr_el_df),
                # TODO what's wrong here, numbers are more than one
                'lossed_of_all': (curr_el_limit_df['TimelimitRaw'] - curr_el_limit_df['ElapsedRaw']).sum()
                                 / src_df['ElapsedRaw'].sum()

            }
        )

time_limit_agg_df = pd.DataFrame(time_limit_agg_df).sort_values(['y_true', 'y_predicted'], ascending=False)
