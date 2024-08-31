import math
import random
import pandas as pd
import numpy as np

from greedy_profit.helpers import Action, datacenters, servers


def get_buy_actions(servers_to_buy: pd.DataFrame, fleet: pd.DataFrame, time_step: int) -> list[Action]:
    """
    Given a table of the number of servers to buy of each generation and latency capacity
    and a table of all servers in the fleet,
    return a list of dictionaries of ideal buy actions.

    The datacenter to buy to is chosen automatically depending on latency sensitivity:
    - 'low' = 'DC1'
    - 'medium' = 'DC2'
    - 'high' = 'DC3' (or 'DC4' if there is not enough space in 'DC3')
    """
    buy_actions: list[Action] = []
    full_datacenters = []
    fleet_with_slots_size = fleet.merge(servers, how='outer', left_on='server_generation', right_on='server_generation')
    current_server_slot_count = fleet_with_slots_size.groupby(by=['datacenter_id'])['slots_size'].sum()
    
    # reverse through the server generations so most valuable are bought first
    for server_generation in reversed(servers_to_buy.index.unique()):
        
        if 'CPU' in server_generation:
            slots_size = 2
        else:
            slots_size = 4
        max_slot_count = datacenters.set_index('datacenter_id')['slots_capacity']
        available_slot_count = max_slot_count - current_server_slot_count

        for latency_sensitivity in servers_to_buy.columns.unique():
            
            if latency_sensitivity == 'low':
                dc = 'DC1'
            elif latency_sensitivity == 'medium':
                dc = 'DC2'
            else:
                dc = 'DC3'
            available_slots = available_slot_count[dc]
            available_servers = available_slots / slots_size

            desired_servers = servers_to_buy.loc[server_generation][latency_sensitivity]
            buy_count = int(min(desired_servers, available_servers))

            # FOR DEBUGGING
            # if buy_count < desired_servers:
            #     full_datacenters.append(dc)

            for _server in range(1, buy_count + 1):
                # TODO: using hashes as IDs for now. Could swap to counting up the number of servers to make dismissing easier
                #       get_server_counts() and system_state.get_servers_in_datacentre() functions would help with this
                buy_actions.append({
                    'action': 'buy',
                    'datacenter_id': dc,
                    'server_generation': server_generation,
                    'server_id': str(hash(random.randbytes(16)))
                })

            # Overflow into DC4
            if latency_sensitivity == 'high' and buy_count < desired_servers:
                # Buy from DC4
                dc4_available_servers = available_slot_count['DC4'] / slots_size
                dc4_count = int(min(desired_servers - buy_count, dc4_available_servers))

                for _server in range(1, dc4_count + 1):
                    buy_actions.append({
                        'action': 'buy',
                        'datacenter_id': 'DC4',
                        'server_generation': server_generation,
                        'server_id': str(hash(random.randbytes(16)))
                    })

                buy_count += dc4_count

            if dc in current_server_slot_count.index.unique():
                current_server_slot_count[dc] += buy_count * slots_size

        # TODO: update the fleet by buying the servers for this generation
        #       ensures the current_server_count is updated for each iteration
    
    # FOR DEBUGGING
    # print(f"Full datacenters: {np.unique(full_datacenters)}")

    return buy_actions