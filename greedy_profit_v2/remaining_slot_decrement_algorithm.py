import numpy as np
import pandas as pd
from greedy_profit_v2.data import break_even_time_all, get_slot_size

def remaining_slots_decrement_algorithm(latency: str, server: str, remaining_slots: pd.DataFrame, desired_buy_count: int, time_step_range: tuple[int, int], use_dc4=False):
    results = []

    slots_size = get_slot_size(server)

    if latency == 'low':
        datacenter_id = 'DC1'
    elif latency == 'medium':
        datacenter_id = 'DC2'
    elif latency == 'high':
        if use_dc4:
            datacenter_id = 'DC4'
        else:
            datacenter_id = 'DC3'
    
    while True:
        # 1) Find the ranges of time steps between which the datacenter can fit at least 1 server
        relevant_remaining_slots = remaining_slots.loc[time_step_range[0]:time_step_range[1]].query(f"{datacenter_id} >= @slots_size")
        time_steps_with_space = relevant_remaining_slots.index.to_numpy()

        if relevant_remaining_slots.empty:
            break

        time_steps_diff = np.diff(time_steps_with_space)
        gap_indices = np.append(np.where(time_steps_diff > 1), len(time_steps_with_space) - 1)

        ranges = []
        start = 0
        for gap in gap_indices:
            ranges.append((time_steps_with_space[start], time_steps_with_space[gap]))
            start = gap + 1


        # 2) Filter ranges which last for less than the break even time
        break_even_time = break_even_time_all[server][latency]
        ranges = [r for r in ranges if r[1] - r[0] >= (2*break_even_time)]


        # 3) For each range (from longest to shortest):
        sorted_ranges_i = np.argsort([r[1] - r[0] for r in ranges])
        for i in reversed(sorted_ranges_i):
            current_range = ranges[i]

            # 1) Calculate the minimum slot capacity
            remaining_slots_in_range = relevant_remaining_slots.loc[current_range[0]:current_range[1]]
            minimum_slot_capacity = remaining_slots_in_range[datacenter_id].min()


            # 2) Pick the minimum of the minimum slot capacity or the number of servers to buy * server slots size
            max_buy_count = int(np.floor(minimum_slot_capacity / slots_size))
            actual_buy_count = min(max_buy_count, desired_buy_count)

            # if actual_buy_count < desired_buy_count:
            #     print(f"{actual_buy_count} servers out of {desired_buy_count} bought, {datacenter_id} is full at {relevant_remaining_slots.query(f"{datacenter_id} == @minimum_slot_capacity")}.index")


            # 3) Store the number of servers to buy, the datacentre, the buy time step, the dismiss time step in the results
            results.append({
                    'server_generation': server,
                    'buy_count': str(actual_buy_count),
                    'datacenter_id': datacenter_id,
                    'buy_time_step': str(current_range[0]),
                    'dismiss_time_step': str(current_range[1] + 1)
                })


            # 4) Subtract the number of bought servers from the initial desired number of servers to buy
            desired_buy_count -= actual_buy_count


            # 5) For each slot capacity in the range for the datacenter, decrease the slot capacity by the result of step 3.2
            actual_slot_count = actual_buy_count * slots_size
            # print(f"Subtracting {actual_slot_count} ({actual_buy_count} servers) from the remaining slots in the {datacenter_id} ({current_range[0]}:{current_range[1]})")
            for index, row in remaining_slots_in_range.iterrows():
                remaining = row[datacenter_id] - actual_slot_count
                remaining_slots.at[index, datacenter_id] = remaining


        # 4) Repeat steps 1 to 3.3 until there are no ranges left after step 2
        if len(ranges) == 0 or desired_buy_count <= 0:
            break

    # 5) Return a tuple of the (results, remaining_slots)
    return results, remaining_slots, desired_buy_count