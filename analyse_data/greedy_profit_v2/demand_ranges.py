import numpy as np
import pandas as pd


def get_demand_ranges(relevant_demand: pd.DataFrame):
    """
    Implements step 1: Find the ranges of time steps between which this server/latency is in demand

    Parameters:
    relevant_demand (pd.DataFrame): the time steps of non-zero demand for a specific server/latency

    Returns a list of tuples of (start, end) time steps for each range of uninterrupted demand.
    """
    time_steps_of_demand = relevant_demand.get('time_step').to_numpy()

    time_steps_diff = np.diff(time_steps_of_demand)
    gap_indices = np.append(np.where(time_steps_diff > 1), len(time_steps_of_demand) - 1)

    ranges = []
    start = 0
    for gap in gap_indices:
        ranges.append((time_steps_of_demand[start], time_steps_of_demand[gap]))
        start = gap + 1

    return ranges

def merge_close_ranges(ranges: list[tuple[int, int]], merge_threshold_multiplier: float):
    """
    Implements step 2: Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)

    Parameters:
    ranges (list[tuple[int, int]]): the list of (start time step, final time step) to merge

    Returns a list of merged ranges.
    """
    i = 0
    while i < len(ranges) - 1:
        length = ranges[i][1] - ranges[i][0]
        length_next = ranges[i + 1][1] - ranges[i + 1][0]

        # ADJUSTABLE
        merge_threshold = min(length, length_next) * merge_threshold_multiplier
        # merge_threshold = 10

        if ranges[i + 1][0] - ranges[i][1] < merge_threshold:
            ranges[i] = (ranges[i][0], ranges[i + 1][1])
            ranges.pop(i + 1)
        else:
            i += 1

    return ranges