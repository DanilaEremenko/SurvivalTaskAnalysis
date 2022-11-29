from datetime import timedelta

TIME_RANGES = {
    '1 (0  - 3   minutes)': (timedelta(minutes=0), timedelta(minutes=3)),
    '2 (3  - 10  minutes)': (timedelta(minutes=3), timedelta(minutes=10)),
    '3 (10 - 60  minutes)': (timedelta(minutes=10), timedelta(hours=1)),
    '4 (1  - 24  hours  )': (timedelta(hours=1), timedelta(days=1)),
    '5 (1  - 3   days   )': (timedelta(days=1), timedelta(days=3)),
    '6 (3  - inf days   )': (timedelta(days=3), timedelta(days=1e5)),
}

TIME_RANGES = {key: (time_range[0].total_seconds(), time_range[1].total_seconds())
               for key, time_range in TIME_RANGES.items()}


def get_time_range_symb(task_time: float) -> str:
    for key, time_range in TIME_RANGES.items():
        if time_range[0] <= task_time < time_range[1]:
            return key

    return 'undefined range'
