import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import uuid

from utils import load_problem_data
from system_state import SystemState

DISMISS_SERVERS_AT_TIME_STEP = 95
LATENCIES = ['high', 'medium', 'low']
SERVER_GENERATIONS = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']

# Mapping of latency sensitivity to datacenters
LATENCY_TO_DC = {
    'low': 'DC1',
    'medium': 'DC2',
    'high': ['DC3', 'DC4']
}


def align_dataframes(srv_counts, demand_srvs_needed):
    # Define the desired columns and row order
    columns = ['latency_sensitivity', 'CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
    row_order = ['high', 'medium', 'low']
    
    # Reshape srv_counts if necessary 
    if isinstance(srv_counts.columns, pd.MultiIndex): # TODO: might not be needed
        srv_counts = srv_counts.droplevel(0, axis=1)
    
    # Ensure 'latency_sensitivity' is a column in srv_counts if it's the index
    if 'latency_sensitivity' not in srv_counts.columns and srv_counts.index.name == 'latency_sensitivity': # TODO: might not be needed
        srv_counts = srv_counts.reset_index()
    
    # Select and reorder columns for both DataFrames
    srv_counts = srv_counts[columns]
    demand_srvs_needed = demand_srvs_needed[columns]
    
    # Set index to latency_sensitivity and reorder rows
    srv_counts = srv_counts.set_index('latency_sensitivity').reindex(row_order)
    demand_srvs_needed = demand_srvs_needed.set_index('latency_sensitivity').reindex(row_order)
    
    return srv_counts, demand_srvs_needed


def init_srv_counts():
    index = pd.MultiIndex.from_product([LATENCIES, SERVER_GENERATIONS], 
                                        names=['latency_sensitivity', 'server_generation'])

    # Initialize srv_counts with zeros
    srv_counts = pd.DataFrame(0, index=index, columns=['count']).unstack(level='server_generation', fill_value=0)
    return srv_counts


def calculate_servers_needed(demand_df: pd.DataFrame, servers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of servers needed to satisfy the demand for all server generations.

    Args:
        demand_df (pd.DataFrame): DataFrame containing demand data for all server generations.
        servers_info (pd.DataFrame): DataFrame containing server information, including capacity.

    Returns:
        pd.DataFrame: Input DataFrame with additional columns 
                      for the number of servers needed for each generation.

    Note:
        The function does not take into account the current fleet. 
        It is intdended to be ran once, before the time steps loop.
    """
    # List of all server generations
    server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']

    # Create a dictionary to store server capacities
    server_capacities = servers_df.set_index('server_generation')['capacity'].to_dict()

    for generation in server_generations:
        # Calculate the number of servers needed for each generation
        # servers_needed_col = f'{generation}_servers_needed'
        number_of_srvs = (demand_df[generation] / server_capacities[generation])
        number_of_srvs = number_of_srvs.apply(np.ceil).astype(int) # np.floor can be potentially more profitable

        demand_df[generation] = number_of_srvs
    return demand_df


def check_fleet_for_old_servers(state: SystemState) -> List[Dict]:
    """
    Check the current fleet for servers that are about to reach their life expectancy 
        and prepare decisions to dismiss them.

    Args:
        state (SystemState): The current state of the system, containing information about the server fleet.

    Returns:
        List[Dict]: A list of dismissal decisions. Each decision is a dictionary with the following keys:
            - 'datacenter_id' (str): The ID of the datacenter where the server is located.
            - 'server_generation' (str): The generation of the server.
            - 'server_id' (str): The unique identifier of the server.
            - 'action' (str): Always set to 'dismiss' for this function.
        
        Returns an empty list if there are no servers that meet the dismissal criteria.

    Note:
        This function assumes that servers have a life expectancy of 96 time steps and targets servers
        at 95 time steps to ensure they are dismissed before reaching their full life expectancy.
    """
    decision = []
    old_servers_to_dismiss = state.fleet.loc[state.fleet['lifespan'] == DISMISS_SERVERS_AT_TIME_STEP]

    if not old_servers_to_dismiss.empty:
        for _, old_server in old_servers_to_dismiss.iterrows():
            decision.append({
                'datacenter_id': old_server['datacenter_id'],
                'server_generation': old_server['server_generation'],
                'server_id': old_server['server_id'],
                'action': 'dismiss'
            })

    return decision


def allocate_servers(state: SystemState, 
                     demand_srvs_needed: pd.DataFrame,
                     srv_counts: pd.DataFrame) -> List[Dict]:
    print(state.solution)
    print(demand_srvs_needed)
    print(srv_counts)
    # First, check and dismiss old servers
    actions = check_fleet_for_old_servers(state) 
    # print(f"Old servers to dismiss: {len(actions)}") 

    # If the fleet is not empty, update srv_counts with actual values
    if not state.fleet.empty:
        srv_counts = state.fleet.groupby(['latency_sensitivity', 'server_generation']).size().unstack(fill_value=0)

    # TODO: Align srv_counts and demand_srvs_needed tructure with each other so I can iterate over them in the same format
    srv_counts, demand_srvs_needed = align_dataframes(srv_counts, demand_srvs_needed)

    # print(f"Current server counts info:\n{srv_counts.info()}")
    # print(f"Current server counts index:\n{srv_counts.index}")
    # print(f"Current server counts columns:\n{srv_counts.columns}")
    # print(srv_counts)

    # print(f"Current demand per server info:\n{demand_srvs_needed.info()}")
    # print(f"Current demand per server info:\n{demand_srvs_needed.index}")
    # print(f"Current demand per server info:\n{demand_srvs_needed.columns}")
    # print(demand_srvs_needed)


    for latency, (_, latency_demand), (_, srvs_per_latency) in zip(LATENCIES, demand_srvs_needed.iterrows(), srv_counts.iterrows()):
        # print(latency_demand)
        # latency = latency_demand['latency_sensitivity']
        # print(f"Processing latency: {latency}")

        srvs_needed = latency_demand.iloc[2:]

        # print(f"Servers needed: {srvs_needed}")
        # print(f"Servers in posession: {srvs_per_latency}")

        srvs_short = np.maximum(0, srvs_needed - srvs_per_latency)
        srvs_excess = np.maximum(0, srvs_per_latency - srvs_needed)

        # print(f"Servers short: {srvs_short}")
        # print(f"Servers excess: {srvs_excess}")

        # TODO: Balance the assets
        # Example 1: 
        #   srvs_short['CPU.S1'] = 4
        #   srvs_excess['CPU.S1'] = 5
        # Such situation means that there are more servers that aren't used 
        # and they can should be moved to datacenters with latency = latency_demand['latency_sensitivity']
        # 
        # Example 2: 
        #   srvs_short['CPU.S1'] = 5
        #   srvs_excess['CPU.S1'] = 4
        # Such situation means that there are some servers that aren't used 
        # they can should be moved to datacenters with latency = latency_demand['latency_sensitivity'],
        # however, more servers needed, so 5 - 4 = 1 server needs to be purchased 
        # for datacenters with latency = latency_demand['latency_sensitivity'].
        # 
        # Important Notes:
        # - When 'high' latency_sensitivity capacity is in excess, DC4 should be prioritized to move from
        # - When 'high' latency_sensitivity capacity is in shortage, DC3 should prioritized to buy for

        # Balance assets
        for server_gen in srvs_short.index:
            if srvs_short[server_gen] > 0:
                # Check if we can move servers from other latencies
                for other_latency in srv_counts.index:
                    if other_latency != latency and srvs_excess[server_gen] > 0:
                        servers_to_move = min(srvs_short[server_gen], srvs_excess[server_gen])
                        
                        # Find servers to move
                        servers_to_move_df = state.fleet[
                            (state.fleet['latency_sensitivity'] == other_latency) &
                            (state.fleet['server_generation'] == server_gen)
                        ].head(servers_to_move)

                        # Determine target datacenter
                        if latency == 'high':
                            # Prioritize DC3 over DC4 for high latency
                            dc3_capacity = state.datacenter_capacity[state.datacenter_capacity['datacenter_id'] == 'DC3']['slots_capacity'].iloc[0]
                            dc3_used = state.datacenter_capacity[state.datacenter_capacity['datacenter_id'] == 'DC3']['used_slots'].iloc[0]
                            target_dc = 'DC3' if (dc3_capacity - dc3_used) >= servers_to_move else 'DC4'
                        else:
                            target_dc = LATENCY_TO_DC[latency]

                        # Add move actions
                        for _, server in servers_to_move_df.iterrows():
                            actions.append({
                                'datacenter_id': target_dc,
                                'server_generation': server_gen,
                                'server_id': server['server_id'],
                                'action': 'move'
                            })

                        # Update shortages and excesses
                        srvs_short[server_gen] -= servers_to_move
                        srvs_excess[server_gen] -= servers_to_move

            # If we still need servers, buy them
            if srvs_short[server_gen] > 0:
                # Determine target datacenter
                if latency == 'high':
                    # Prioritize DC3 over DC4 for high latency
                    dc3_capacity = state.datacenter_capacity[state.datacenter_capacity['datacenter_id'] == 'DC3']['slots_capacity'].iloc[0]
                    dc3_used = state.datacenter_capacity[state.datacenter_capacity['datacenter_id'] == 'DC3']['used_slots'].iloc[0]
                    target_dc = 'DC3' if (dc3_capacity - dc3_used) >= srvs_short[server_gen] else 'DC4'
                else:
                    target_dc = LATENCY_TO_DC[latency]

                # Add buy actions
                for _ in range(int(srvs_short[server_gen])):
                    actions.append({
                        'datacenter_id': target_dc,
                        'server_generation': server_gen,
                        'server_id': str(uuid.uuid4()),
                        'action': 'buy'
                    })
    # print(f"Total actions: {len(actions)}")
    return actions



def get_solution(datacenters: pd.DataFrame, 
                 servers: pd.DataFrame, 
                 demand: pd.DataFrame) -> List[Dict]:
    state = SystemState(datacenters, servers)

    demand_srvs_needed = calculate_servers_needed(demand, servers)
    srv_counts = init_srv_counts()
    
    for ts in range(1, 169):  # 168 time steps
        current_demand_srvs_needed = demand_srvs_needed.loc[demand['time_step'] == ts]
        actions = allocate_servers(state, 
                                   current_demand_srvs_needed,
                                   srv_counts)
        
        state.update_state(actions)
    
    return state.solution


def main():
    demand, datacenters, servers, selling_prices = load_problem_data()
    solution = get_solution(datacenters, servers, demand)
    
    # Convert solution to DataFrame and save as JSON
    solution_df = pd.DataFrame(solution)
    solution_df.to_json('./data/solution.json', orient='records', indent=4)

if __name__ == "__main__":
    main()

