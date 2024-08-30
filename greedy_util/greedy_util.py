import pandas as pd
import numpy as np

from typing import List, Dict, Set, Tuple
import uuid
import sys

# setting path
sys.path.append('..')

from utils import load_problem_data
from evaluation import get_actual_demand
from system_state import SystemState

DISMISS_SERVERS_AT_TIME_STEP = 95
LATENCIES = ['high', 'medium', 'low']
SERVER_GENERATIONS = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']

# Mapping of latency sensitivity to datacenters
LATENCY_TO_DC = {
    'low': ['DC1'],
    'medium': ['DC2'],
    'high': ['DC3', 'DC4']
}

def get_target_dc(latency, server_slots, datacenters_cap):
    available_dcs = []
    for dc in LATENCY_TO_DC[latency]:
        dc_info = datacenters_cap[datacenters_cap['datacenter_id'] == dc].iloc[0]
        available_slots = dc_info['slots_capacity'] - dc_info['used_slots']
        if available_slots >= server_slots:
            available_dcs.append((dc, available_slots))
    
    if not available_dcs:
        return None
    
    # Choose the DC with the most available slots
    return max(available_dcs, key=lambda x: x[1])[0]

def calculate_servers(demand_df: pd.DataFrame, servers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of servers needed to satisfy the demand for all server generations.
    Args:
        demand_df (pd.DataFrame): DataFrame containing demand data for all server generations.
        servers_df (pd.DataFrame): DataFrame containing server information, including capacity.
    Returns:
        pd.DataFrame: Input DataFrame with number of servers needed instead of capacity
    Note:
        The function does not take into account the current fleet. 
        It is intended to be ran once, before the time steps loop.
    """
    # Merge demand_df with servers_df to get capacity information
    merged_df = pd.merge(demand_df, servers_df[['server_generation', 'capacity']], on='server_generation', how='left')
    
    # Calculate servers needed for each demand scenario
    for scenario in LATENCIES:
        merged_df[f'{scenario}_servers'] = np.ceil(merged_df[scenario] / merged_df['capacity']).astype(int)
    
    # Drop the capacity column and rename the new columns
    result_df = merged_df.drop(columns=['capacity'] + LATENCIES)
    result_df = result_df.rename(columns={
        'high_servers': 'high',
        'medium_servers': 'medium',
        'low_servers': 'low'
    })
    
    return result_df


def srvs_count(fleet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count servers in the fleet deployment by server generation and datacenter latency.
    
    Args:
        fleet_df (pd.DataFrame): DataFrame containing fleet information.
    
    Returns:
        pd.DataFrame: Summary of servers deployed by generation and latency sensitivity.
    """
    # Group by server generation and latency sensitivity, then count the servers
    srvs_count = fleet_df.groupby(['server_generation', 'latency_sensitivity']).size().unstack(fill_value=0)
    
    # Ensure all required columns are present, add them with zeros if missing
    for col in LATENCIES:
        if col not in srvs_count.columns:
            srvs_count[col] = 0
    
    return srvs_count[LATENCIES].reset_index().sort_values('server_generation')


def calculate_demand_def_exc(srvs_count: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the difference between deployed servers and demand for each server generation and latency.
    
    Args:
        srvs_count (pd.DataFrame): Summary of deployed servers by generation and latency.
        demand (pd.DataFrame): Demand for servers by generation and latency.
    
    Returns:
        pd.DataFrame: Difference between deployed servers and demand. 
                      Positive values indicate excess capacity,
                      negative values indicate unmet demand.
    """
    # Step 1: Create a dataframe with all server generations and initialize with zeros
    # all_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
    result = pd.DataFrame({
        'server_generation': SERVER_GENERATIONS,
        'high': 0,
        'medium': 0,
        'low': 0
    })
    
    # Step 2: Update the result with fleet summary data
    for _, row in srvs_count.iterrows():
        gen = row['server_generation']

        if gen in result['server_generation'].values:
            mask = result['server_generation'] == gen

            for col in LATENCIES:
                result.loc[mask, col] = row[col]
    
    # Step 3: Subtract demand from the result
    for _, row in demand.iterrows():
        gen = row['server_generation']
        
        if gen in result['server_generation'].values:
            mask = result['server_generation'] == gen
            
            for col in LATENCIES:
                result.loc[mask, col] = result.loc[mask, col].values - row[col]
    
    return result


def check_fleet_for_old_servers(state: SystemState) -> Tuple[List, pd.DataFrame, Set]:
    """
    Check the current fleet for servers that are about to reach their life expectancy 
        and prepare actions to dismiss them.

    Args:
        state (SystemState): The current state of the system, 
                             containing information about the server fleet.

    Returns:
        list: A list of dismissal actions. Each action is a dictionary with the following keys:
            - 'datacenter_id' (str): The ID of the datacenter where the server is located.
            - 'server_generation' (str): The generation of the server.
            - 'server_id' (str): The unique identifier of the server.
            - 'action' (str): Always set to 'dismiss' for this function.
        
        Returns an empty list if there are no servers that meet the dismissal criteria.
        set: A set of processed servers

    Note:
        This function assumes that servers have a life expectancy of 96 time steps and targets servers
        at 95 time steps to ensure they are dismissed before reaching their full life expectancy.
    """
    actions = []
    old_servers_to_dismiss = state.fleet.loc[state.fleet['lifespan'] == DISMISS_SERVERS_AT_TIME_STEP - 1]

    # Set to keep track of servers that have been processed (moved, bought, or dismissed)
    processed_servers = set()

    if not old_servers_to_dismiss.empty:
        for _, old_server in old_servers_to_dismiss.iterrows():
            actions.append({
                'datacenter_id': old_server['datacenter_id'],
                'server_generation': old_server['server_generation'],
                'server_id': old_server['server_id'],
                'action': 'dismiss',
                'dismissed_by': 'check_fleet_for_old_servers'
            })

            processed_servers.add(old_server['server_id'])

    return actions, processed_servers


def move_servers(state: SystemState, 
                 demand_def_exc: pd.DataFrame, 
                 actions: List,
                 processed_servers: Set) -> Tuple[List, pd.DataFrame, Set]:
    """
    Move servers between datacenters to satisfy demand based on calculated demand deficit and excess.
    
    Args:
        state (SystemState): The current state of the system, 
                             containing information about the server fleet
                             and datacenters' available space.
        demand_def_exc (pd.DataFrame): DataFrame with demand deficit and excess calculations.
        set: A set of processed servers
        
    Returns:
        list: List of dictionaries containing server movement actions.
        pd.DataFrame: updated demand_def_exc to find which demand yet to satisfy
        set: updated set of processed servers
    """
    
    fleet_df = state.fleet
    datacenters_df = state.datacenter_capacity.copy()
    updated_demand_def_exc = demand_def_exc.copy()

    for idx, row in demand_def_exc.iterrows():
        server_generation = row['server_generation']
        server_slots = state.servers_info.loc[state.servers_info['server_generation'] == server_generation, 'slots_size'].iloc[0]

        for from_latency in LATENCIES:
            excess = row[from_latency]

            if excess <= 0:
                continue

            for to_latency in LATENCIES:
                deficit = -row[to_latency]

                if deficit <= 0:
                    continue

                servers_to_move = min(excess, deficit)

                servers = fleet_df[(fleet_df['server_generation'] == server_generation) &
                                   (fleet_df['datacenter_id'].isin(LATENCY_TO_DC[from_latency])) &
                                   (~fleet_df['server_id'].isin(processed_servers))]  # Exclude already processed servers

                for _, server in servers.iterrows():
                    if servers_to_move == 0:
                        break

                    target_dc = get_target_dc(to_latency, server_slots, datacenters_df)

                    # Check for full DC
                    dc_mask = datacenters_df['datacenter_id'] == target_dc
                    dc_fullness = datacenters_df.loc[dc_mask, 'used_slots'] >= datacenters_df.loc[dc_mask, 'slots_capacity']
                    if target_dc is None or dc_fullness.any():
                        continue  # No space in target datacenters

                    # Update datacenter used slots

                    datacenters_df.loc[dc_mask, 'used_slots'] += server_slots

                    actions.append({
                        'datacenter_id': target_dc,
                        'server_generation': server_generation,
                        'server_id': server['server_id'],
                        'action': 'move'
                    })

                    # Add the server to the set of moved servers
                    processed_servers.add(server['server_id'])

                    servers_to_move -= 1
                    excess -= 1
                    
                    updated_demand_def_exc.at[idx, from_latency] -= 1
                    updated_demand_def_exc.at[idx, to_latency] += 1

                if excess == 0:
                    break

            if excess == 0:
                break

    return actions, updated_demand_def_exc, processed_servers


def buy_servers(state: SystemState, 
                demand_def_exc: pd.DataFrame, 
                actions: List,
                processed_servers: Set)-> Tuple[List, pd.DataFrame, Set]:
    """
    Determine if new servers need to be bought to satisfy remaining demand 
        and add 'buy' actions if necessary.
    
    Args:
        demand_def_exc (pd.DataFrame): DataFrame with 
                   current demand deficit and excess calculations.
        actions (list): List of existing actions (moves, etc.).
    
    Returns:
        list: Updated list of actions, including new 'buy' actions if needed.
        pd.DataFrame: updated demand_def_exc to find which servers are not utilized
        set: updated set of processed servers 
    """
    datacenters_df = state.datacenter_capacity.copy()
    updated_demand_def_exc = demand_def_exc.copy()
    
    for idx, row in demand_def_exc.iterrows():
        server_generation = row['server_generation']
        server_slots = state.servers_info.loc[state.servers_info['server_generation'] == server_generation, 'slots_size'].iloc[0]
        
        for latency in LATENCIES:
            deficit = max(0, -row[latency])  # Ensure we only consider negative values as deficit
            
            if deficit > 0:
                for _ in range(deficit):
                    target_dc = get_target_dc(latency, server_slots, datacenters_df)
                    dc_mask = datacenters_df['datacenter_id'] == target_dc
                    dc_fullness = datacenters_df.loc[dc_mask, 'used_slots'] >= datacenters_df.loc[dc_mask, 'slots_capacity']
                    if target_dc is None or dc_fullness.any():
                        # print(f"Warning: No available datacenter for {latency} latency. Skipping buy action.")
                        continue
                
                    # Update datacenter used slots
                    datacenters_df.loc[dc_mask, 'used_slots'] += server_slots

                    server_id = str(uuid.uuid4())
                    actions.append({
                        'datacenter_id': target_dc,
                        'server_generation': server_generation,
                        'server_id': server_id,
                        'action': 'buy'
                    })

                    processed_servers.add(server_id)
                    updated_demand_def_exc.at[idx, latency] += 1
    
    return actions, updated_demand_def_exc, processed_servers

def dismiss_servers(state: SystemState, 
                    demand_def_exc: pd.DataFrame, 
                    actions: List,
                    processed_servers: Set) -> List:
    
    fleet_df = state.fleet
    datacenters_df = state.datacenter_capacity.copy()

    for _, row in demand_def_exc.iterrows():
        server_generation = row['server_generation']
        server_slots = state.servers_info.loc[state.servers_info['server_generation'] == server_generation, 'slots_size'].iloc[0]
        
        for latency in LATENCIES:
            excess = max(0, row[latency])  # Ensure we only consider positive values as excess
            
            if excess > 0:

                servers = fleet_df[(fleet_df['server_generation'] == server_generation) &
                                   (fleet_df['datacenter_id'].isin(LATENCY_TO_DC[latency])) &
                                   (~fleet_df['server_id'].isin(processed_servers))]  # Exclude already processed servers

                for _, server in servers.iterrows():
                    target_dc = get_target_dc(latency, server_slots, datacenters_df)
                    
                    if target_dc is None or excess == 0:
                        # print(f"Warning: No available datacenter for {latency} latency. Skipping buy action.")
                        continue
                
                    # Update datacenter used slots
                    dc_mask = datacenters_df['datacenter_id'] == target_dc
                    datacenters_df.loc[dc_mask, 'used_slots'] -= server_slots

                    actions.append({
                        'datacenter_id': target_dc,
                        'server_generation': server_generation,
                        'server_id': server['server_id'],
                        'action': 'dismiss',
                        'dismissed_by': 'dismiss_servers'
                    })

                    processed_servers.add(server['server_id'])
                    excess -= 1

    return actions


def allocate_servers(state: SystemState, demand: pd.DataFrame) -> List[Dict]:
    # Dismiss old servers
    actions, processed_servers = check_fleet_for_old_servers(state)
    state.update_fleet(actions)
    print('in check_fleet_for_old_servers')
    state.update_datacenter_capacity()
    
    srvs_cnt = srvs_count(state.fleet)
    demand_def_exc = calculate_demand_def_exc(srvs_cnt, demand)

    # Move servers
    actions, updated_demand_def_exc, processed_servers = move_servers(state, 
                                                                      demand_def_exc,
                                                                      actions, 
                                                                      processed_servers)
    state.update_fleet(actions)
    print('in move_servers')
    state.update_datacenter_capacity()

    # Buy servers
    actions, updated_demand_def_exc, processed_servers = buy_servers(state, 
                                                                     updated_demand_def_exc, 
                                                                     actions,
                                                                     processed_servers)
    state.update_fleet(actions)
    print('in buy_servers')
    state.update_datacenter_capacity()

    # Dismiss unused servers
    actions = dismiss_servers(state, 
                              updated_demand_def_exc, 
                              actions,
                              processed_servers)

    return actions


def get_solution(datacenters: pd.DataFrame, 
                 servers: pd.DataFrame, 
                 demand: pd.DataFrame) -> List[Dict]:
    state = SystemState(datacenters, servers)

    demand_srvs = calculate_servers(demand, servers)
    
    for ts in range(1, 169):  # 168 time steps
        print(ts)
        current_demand_srvs = demand_srvs.loc[demand['time_step'] == ts]
        actions = allocate_servers(state, current_demand_srvs)

        print('in dismiss_servers')
        state.update_state(actions)
        print(state.datacenter_capacity)

        if ts == 35:
            fleet = state.fleet
            # fleet.drop_duplicates('server_id', inplace=False)
            fleet.set_index('server_id', drop=False, inplace=True)
            fleet.to_json('fleet_at_35.json')
    
    return state.solution


def main():
    np.random.seed(1741)

    demand, datacenters, servers, _ = load_problem_data('../data')
    actual_demand = get_actual_demand(demand)
    solution = get_solution(datacenters, servers, actual_demand)
    
    # Convert solution to DataFrame and save as JSON
    solution_df = pd.DataFrame(solution)
    solution_df.to_json('../data/solution.json', orient='records', indent=4)

if __name__ == "__main__":
    main()

