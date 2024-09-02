import numpy as np
import pandas as pd
from greedy_profit_v2.data import get_slot_size, get_sorted_servers, break_even_time_all, servers
from greedy_profit_v2.demand_ranges import get_demand_ranges, merge_close_ranges


def greedy_profit_algorithm(actual_demand: pd.DataFrame):
    results = []

    # 1) For each server/latency combination (in order of profitability):
    sorted_servers = get_sorted_servers('data/test_data/most_profitable_servers_by_artem.csv')
    for server_generation, latency_sensitivity in sorted_servers:
        remaining_demand = actual_demand.copy()
        slots_size = get_slot_size(server_generation)

        # TEMPORARY
        if server_generation in ['CPU.S1', 'CPU.S2']:
            continue

        if latency_sensitivity == 'low':
            datacenter_id = 'DC1'
        elif latency_sensitivity == 'medium':
            datacenter_id = 'DC2'
        elif latency_sensitivity == 'high':
            datacenter_id = 'DC3'

        print(f"Server generation: {server_generation}, Latency sensitivity: {latency_sensitivity}")
        while True:
            # 1) Find the ranges of time steps between which this server/latency is in demand
            relevant_demand = remaining_demand.query(f'server_generation == @server_generation and {latency_sensitivity} > 0')
            ranges = get_demand_ranges(relevant_demand)

            # 2) Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)
            ranges = merge_close_ranges(ranges)

            # 3) Filter all ranges which last for less than the time it takes for the server/latency to break even
            break_even_time = break_even_time_all[server_generation][latency_sensitivity]
            # ADJUSTABLE
            minimum_range_length = break_even_time * 2
            ranges = [r for r in ranges if r[1] - r[0] >= minimum_range_length]

            
            # 4) For each range (from longest to shortest):
            sorted_ranges_i = np.argsort([r[1] - r[0] for r in ranges])
            for i in reversed(sorted_ranges_i):
                current_range = ranges[i]

                # 1) Calculate the minimum demand across that range
                demand_in_range = relevant_demand.query(f'time_step >= @current_range[0] and time_step <= @current_range[1]')
                min_demand = demand_in_range.min()[latency_sensitivity]


                # 2) Calculate the number of servers to buy meet the minimum demand
                capacity = servers.set_index('server_generation').loc[server_generation]['capacity']
                desired_buy_count = int(np.round(min_demand / capacity))
                # print(f"{min_demand}/{capacity} = {min_demand / capacity} ~~ {str(desired_buy_count)} GPUs to buy")


                # 3) Store the number of servers to buy, which data centre, the buy time step, the dismiss time step
                results.append({
                    'server_generation': server_generation,
                    'buy_count': str(desired_buy_count),
                    'datacenter_id': datacenter_id,
                    'buy_time_step': str(current_range[0]),
                    'dismiss_time_step': str(current_range[1] + 1)
                })


                # 4) For each demand in the range, subtract the capacity * number of servers to buy
                demand_to_subtract = desired_buy_count * capacity
                # print(f"Subtracting {demand_to_subtract} from the demand in the range")
                for index, row in demand_in_range.iterrows():
                    remaining = row[latency_sensitivity] - demand_to_subtract
                    remaining_demand.at[index, latency_sensitivity] = remaining


                # 5) Filter new demand values which are too low to buy at least 1 server for
                remaining_demand = remaining_demand.query(f'{latency_sensitivity} > {(capacity / 2) + 1}')


            # 5) Repeat steps 1.1 to 1.4.5 with the new demand values until there are no ranges after 1.2
            if len(ranges) == 0:
                results_df = pd.DataFrame(results)
                total_servers_bought = results_df['buy_count'].astype(int).sum()
                print(f"Total servers bought: {total_servers_bought}")
                break
    
    return results