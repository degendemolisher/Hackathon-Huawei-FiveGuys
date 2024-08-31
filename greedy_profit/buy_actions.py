import math
import random
import pandas as pd

from helpers import Action, datacenters, servers


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
    
    # reverse through the server generations so most valuable are bought first
    for server_generation in reversed(servers_to_buy.index.unique()):
        
        if 'CPU' in server_generation:
            slot_size = 2
        else:
            slot_size = 4
        max_buy_count = datacenters.set_index('datacenter_id')['slots_capacity'].map(lambda x: math.floor(x / slot_size))
        current_server_count = fleet.groupby(by=['datacenter_id'])['server_id'].count()
        available_buy_count = max_buy_count - current_server_count

        for latency_sensitivty in servers_to_buy.columns.unique():
            
            # TODO:  WIP validate there is enough space in the datacenter for the servers
            #       /update n to be the maximum number of servers you can buy before the DC is full
            if latency_sensitivty == 'low':
                dc = 'DC1'
            elif latency_sensitivty == 'medium':
                dc = 'DC2'
            else:
                dc = 'DC3'
            available = available_buy_count[dc]

            n = servers_to_buy.loc[server_generation][latency_sensitivty]
            buy_count = max(n, available)

            for _server in range(1, n + 1):
                # TODO: using hashes as IDs for now. Could swap to counting up the number of servers to make dismissing easier
                #       get_server_counts() and system_state.get_servers_in_datacentre() functions would help with this
                buy_actions.append({
                    'action': 'buy',
                    'datacenter_id': dc,
                    'server_generation': server_generation,
                    'time_step': time_step,
                    'server_id': hash(random.randbytes(16))
                })

            # Overflow into DC4
            if latency_sensitivty == 'high' and buy_count < n:
                # Buy from DC4
                dc4_count = buy_count - n

                for _server in range(1, dc4_count + 1):
                    buy_actions.append({
                        'action': 'buy',
                        'datacenter_id': 'DC4',
                        'server_generation': server_generation,
                        'time_step': time_step,
                        'server_id': hash(random.randbytes(16))
                    })

        # TODO: update the fleet by buying the servers for this generation
        #       ensures the current_server_count is updated for each iteration
    
    return buy_actions