import numpy as np
import pandas as pd
from greedy_profit_v2.data import get_slot_size, get_sorted_servers, break_even_time_all, servers
from greedy_profit_v2.demand_ranges import get_demand_ranges, merge_close_ranges
from greedy_profit_v2.remaining_slot_decrement_algorithm import remaining_slots_decrement_algorithm


def greedy_profit_algorithm(actual_demand: pd.DataFrame):
    results = []

    # 1) Initialise a DataFrame that tracks the remaining slots of each datacentre at each time step
    datacenter_slot_capacities = { 'DC1': 25245, 'DC2': 15300, 'DC3': 7020, 'DC4': 8280 }
    remaining_slots = pd.DataFrame(datacenter_slot_capacities, index=range(1, 169), columns=['DC1', 'DC2', 'DC3', 'DC4']).rename_axis('time_step')
    # print(remaining_slots)

    # 2) For each server/latency combination (in order of profitability):
    sorted_servers = get_sorted_servers('data/test_data/most_profitable_servers_by_artem.csv')
    for server_generation, latency_sensitivity in sorted_servers:
        remaining_demand = actual_demand.copy()


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


                # 3) Perform the Remaining Slot Decrement Algorithm
                validated_results, remaining_slots, leftover_buy_count = remaining_slots_decrement_algorithm(latency_sensitivity, server_generation, remaining_slots, desired_buy_count, current_range)
                results.extend(validated_results)

                # 3.1) If there are leftover serveers to buy and the latency is high, perform the Remaining Slot Decrement Algorithm with DC4
                if latency_sensitivity == 'high' and leftover_buy_count > 0:
                    dc4_results, remaining_slots, _dc4_leftover_buy_count = remaining_slots_decrement_algorithm(latency_sensitivity, server_generation, remaining_slots, leftover_buy_count, current_range, use_dc4=True)
                    results.extend(dc4_results)

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
                print(f"{server_generation}, {latency_sensitivity}: Total servers bought: {total_servers_bought}")
                break
    
    # DEBUG: Save remaining slots to CSV
    remaining_slots.to_csv('./remaining_slots.csv', sep='\t', encoding='utf-8')

    return results