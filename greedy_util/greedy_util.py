import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import List, Dict, Set, Tuple
import uuid
import sys

# setting path
sys.path.append('..')

from utils import load_problem_data
from evaluation import get_actual_demand, get_known
from system_state import SystemState

DEMAND, DATACENTERS, SERVERS, _ = load_problem_data('../data')

TEST_SEED = 1741
MA_WINDOW_SIZE = 24
DISMISS_SERVERS_AT_TIME_STEP = 95

TOTAL_TIME_STEPS = get_known('time_steps')
LATENCIES = get_known('latency_sensitivity')
SERVER_GENERATIONS = get_known('server_generation')

# Mapping of server generation to slots size
SLOTS_SIZE_MAP = SERVERS.set_index('server_generation')['slots_size']

# Mapping of latency sensitivity to datacenters
LATENCY_TO_DC = {
    'low': ['DC1'],
    'medium': ['DC2'],
    'high': ['DC3', 'DC4']
}


def calculate_moving_average(actual_demand, window_size=6):
    # Group by server_generation
    grouped = actual_demand.groupby('server_generation')
    
    # Calculate moving average for each group
    ma_dfs = []

    for _, group in grouped:
        ma_df = group.copy()

        for col in ['high', 'low', 'medium']:
            ma_df[col] = group[col].rolling(window=window_size, min_periods=1).mean().round().astype(int)

        ma_dfs.append(ma_df)
    
    # Combine all moving average dataframes
    ma_actual_demand = pd.concat(ma_dfs).sort_index()
    return ma_actual_demand


def calculate_servers(actual_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of servers needed to satisfy the demand for all server generations.
    Args:
        actual_demand (pd.DataFrame): DataFrame containing demand data for all server generations.
    Returns:
        pd.DataFrame: Input DataFrame with number of servers needed instead of capacity
    Note:
        The function does not take into account the current fleet. 
        It is intended to be ran once, before the time steps loop.
    """
    # Merge actual_demand with SERVERS to get capacity information
    merged_df = pd.merge(actual_demand, SERVERS[['server_generation', 'capacity']], on='server_generation', how='left')
    
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
                'action': 'dismiss'
            })

            processed_servers.add(old_server['server_id'])

    return actions, processed_servers


def move_servers(state: SystemState, 
                 demand_def_exc: pd.DataFrame, 
                 processed_servers: Set) -> Tuple[List[Dict], pd.DataFrame, Set]:
    """
    Move servers between datacenters to satisfy demand based on calculated demand deficit and excess.

    Args:
        state (SystemState): The current state of the system, 
                             containing information about the server fleet
                             and datacenters' available space.
        demand_def_exc (pd.DataFrame): DataFrame with demand deficit and excess calculations.
        processed_servers (Set): A set of servers that have already been processed.

    Returns:
        Tuple[List[Dict], pd.DataFrame, Set]: 
            - List of dictionaries containing server movement actions.
            - Updated demand_def_exc to reflect satisfied demand.
            - Updated set of processed servers.
    """
    actions = []
    fleet_df = state.fleet
    datacenters_df = state.datacenter_capacity.copy()
    updated_demand_def_exc = demand_def_exc.copy()

    # Convert demand data to long format for easier processing
    long_demand = _melt_demand_dataframe(demand_def_exc)
    excess_servers, deficit_servers = _split_demand(long_demand)

    for excess, deficit in zip(excess_servers.itertuples(), deficit_servers.itertuples()):
        if excess.server_generation != deficit.server_generation:
            continue

        server_generation = excess.server_generation

        from_latency, to_latency = excess.latency, deficit.latency
        servers_to_move = min(excess.demand, -deficit.demand)   # Number of servers to move
        server_slots = SLOTS_SIZE_MAP[server_generation]

        available_servers = _find_available_servers(fleet_df, server_generation, from_latency, processed_servers)
        target_dcs = _find_target_datacenters(datacenters_df, DATACENTERS, to_latency, server_slots)

        if target_dcs.empty or available_servers.empty:
            continue

        actions, processed_servers, updated_demand_def_exc = _process_server_moves(
            available_servers, target_dcs, servers_to_move, server_generation, server_slots,
            from_latency, to_latency, actions, processed_servers, updated_demand_def_exc, datacenters_df
        )

    return actions, updated_demand_def_exc, processed_servers


def buy_servers(state: SystemState, 
                demand_def_exc: pd.DataFrame, 
                processed_servers: Set) -> Tuple[List[Dict], pd.DataFrame, Set]:
    """
    Buy new servers to satisfy demand deficit.

    Args:
        state (SystemState): The current state of the system.
        demand_def_exc (pd.DataFrame): DataFrame with demand deficit and excess calculations.
        processed_servers (Set): A set of servers that have already been processed.

    Returns:
        Tuple[List[Dict], pd.DataFrame, Set]: 
            - List of dictionaries containing server purchase actions.
            - Updated demand_def_exc to reflect satisfied demand.
            - Updated set of processed servers.
    """
    actions = []
    datacenters_df = state.datacenter_capacity.copy()
    updated_demand_def_exc = demand_def_exc.copy()

    long_demand = _melt_demand_dataframe(demand_def_exc)
    deficit_servers = _get_deficit_servers(long_demand)

    for deficit in deficit_servers.itertuples():
        server_generation = deficit.server_generation
        latency = deficit.latency
        server_slots = SLOTS_SIZE_MAP[server_generation]

        target_dcs = _find_target_datacenters(datacenters_df, DATACENTERS, latency, server_slots)
        if target_dcs.empty:
            continue
        
        # Calculate how many servers can be bought for each datacenter
        target_dcs['can_buy'] = ((target_dcs['slots_capacity'] - target_dcs['used_slots']) // server_slots).clip(upper=deficit.deficit_amount)

        actions, processed_servers, updated_demand_def_exc, datacenters_df = _process_server_purchases(
            target_dcs, deficit.deficit_amount, server_generation, server_slots, latency,
            actions, processed_servers, updated_demand_def_exc, datacenters_df
        )

    return actions, updated_demand_def_exc, processed_servers


def dismiss_servers(state: SystemState, 
                    demand_def_exc: pd.DataFrame, 
                    processed_servers: Set) -> List[Dict]:
    """
    Dismiss excess servers to optimise resource allocation based on demand excess.

    Args:
        state (SystemState): The current state of the system.
        demand_def_exc (pd.DataFrame): DataFrame with demand deficit and excess calculations.
        processed_servers (Set): A set of servers that have already been processed.

    Returns:
        List[Dict]: List of dictionaries containing server dismissal actions.
    """
    actions = []
    fleet_df = state.fleet
    datacenters_df = state.datacenter_capacity.copy()

    long_demand = _melt_demand_dataframe(demand_def_exc)
    excess_servers = _get_excess_servers(long_demand)

    for excess in excess_servers.itertuples():
        server_generation = excess.server_generation
        latency = excess.latency
        server_slots = SLOTS_SIZE_MAP[server_generation]

        dismissable_servers = _find_dismissable_servers(fleet_df, server_generation, latency, processed_servers)
        if dismissable_servers.empty:
            continue

        dc_costs = _get_sorted_dc_costs(DATACENTERS, latency)

        actions, processed_servers, datacenters_df = _process_server_dismissals(
            dc_costs, dismissable_servers, excess.excess_amount, server_generation, server_slots,
            actions, processed_servers, datacenters_df
        )

    return actions


# Helper functions
def _melt_demand_dataframe(demand_def_exc: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the demand_def_exc DataFrame from wide to long format.
    """
    return demand_def_exc.melt(id_vars=['server_generation'], 
                               var_name='latency', 
                               value_name='demand')


def _split_demand(long_demand: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the demand DataFrame into excess and deficit servers.
    """
    excess_servers = long_demand[long_demand['demand'] > 0]
    deficit_servers = long_demand[long_demand['demand'] < 0]
    return excess_servers, deficit_servers


def _find_available_servers(fleet_df: pd.DataFrame, server_generation: str, 
                            latency: str, processed_servers: Set) -> pd.DataFrame:
    """
    Find available servers for a given server generation and latency that haven't been processed yet.
    """
    return fleet_df[
        (fleet_df['server_generation'] == server_generation) &
        (fleet_df['datacenter_id'].isin(LATENCY_TO_DC[latency])) &
        (~fleet_df['server_id'].isin(processed_servers))
    ]


def _find_target_datacenters(datacenters_df: pd.DataFrame, DATACENTERS: pd.DataFrame, 
                             latency: str, server_slots: int) -> pd.DataFrame:
    """
    Find target datacenters that can accommodate servers for a given latency and server size.
    """
    target_dcs = datacenters_df[
        (datacenters_df['datacenter_id'].isin(LATENCY_TO_DC[latency])) &
        (datacenters_df['slots_capacity'] - datacenters_df['used_slots'] >= server_slots)
    ]

    # Merge with DATACENTERS to get cost_of_energy and sort by it
    target_dcs = target_dcs.merge(DATACENTERS[['datacenter_id', 'cost_of_energy']], on='datacenter_id')
    return target_dcs.sort_values('cost_of_energy')


def _process_server_moves(available_servers: pd.DataFrame, target_dcs: pd.DataFrame, 
                          servers_to_move: int, server_generation: str, server_slots: int,
                          from_latency: str, to_latency: str, actions: List[Dict], 
                          processed_servers: Set, updated_demand_def_exc: pd.DataFrame, 
                          datacenters_df: pd.DataFrame) -> Tuple[List[Dict], Set, pd.DataFrame]:
    """
    Process server moves from one datacenter to another.
    """
    for _, server in available_servers.iterrows():
        if servers_to_move == 0 or target_dcs.empty:
            break

        target_dc = target_dcs.iloc[0]['datacenter_id']

        actions.append({
            'datacenter_id': target_dc,
            'server_generation': server_generation,
            'server_id': server['server_id'],
            'action': 'move'
        })

        processed_servers.add(server['server_id'])

        # Update datacenter capacity
        datacenters_df.loc[datacenters_df['datacenter_id'] == target_dc, 'used_slots'] += server_slots
        target_dcs.loc[target_dcs['datacenter_id'] == target_dc, 'used_slots'] += server_slots

        # Remove full datacenters from consideration
        target_dcs = target_dcs[target_dcs['slots_capacity'] - target_dcs['used_slots'] >= server_slots]

        # Update demand deficit/excess
        updated_demand_def_exc.loc[
            updated_demand_def_exc['server_generation'] == server_generation, 
            [from_latency, to_latency]
        ] += [-1, 1]

        servers_to_move -= 1

    return actions, processed_servers, updated_demand_def_exc


def _get_deficit_servers(long_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Get servers with demand deficit and sort them by deficit amount.
    """
    deficit_servers = long_demand[long_demand['demand'] < 0].copy()

    # Convert negative demand to positive deficit
    deficit_servers['deficit_amount'] = -deficit_servers['demand']
    return deficit_servers.sort_values('deficit_amount', ascending=False)


def _get_excess_servers(long_demand: pd.DataFrame) -> pd.DataFrame:
    
    excess_servers = long_demand[long_demand['demand'] > 0].copy()
    excess_servers = excess_servers.rename(columns={'demand': 'excess_amount'})
    return excess_servers.sort_values('excess_amount', ascending=False)


def _process_server_purchases(target_dcs: pd.DataFrame, deficit: int, server_generation: str, 
                              server_slots: int, latency: str, actions: List[Dict], 
                              processed_servers: Set, updated_demand_def_exc: pd.DataFrame, 
                              datacenters_df: pd.DataFrame) -> Tuple[List[Dict], Set, pd.DataFrame, pd.DataFrame]:
    """
    Process server purchases for datacenters with capacity.
    """
    for _, dc in target_dcs.iterrows():
        if deficit == 0:
            break

        to_buy = min(dc['can_buy'], deficit)

        for _ in range(to_buy):
            server_id = str(uuid.uuid4())
            actions.append({
                'datacenter_id': dc['datacenter_id'],
                'server_generation': server_generation,
                'server_id': server_id,
                'action': 'buy'
            })

            processed_servers.add(server_id)

        datacenters_df.loc[datacenters_df['datacenter_id'] == dc['datacenter_id'], 'used_slots'] += to_buy * server_slots
        updated_demand_def_exc.loc[
            updated_demand_def_exc['server_generation'] == server_generation, 
            latency
        ] += to_buy

        deficit -= to_buy

    return actions, processed_servers, updated_demand_def_exc, datacenters_df


def _find_dismissable_servers(fleet_df: pd.DataFrame, server_generation: str, 
                              latency: str, processed_servers: Set) -> pd.DataFrame:
    """
    Find servers that can be dismissed for a given server generation and latency.
    """
    return fleet_df[
        (fleet_df['server_generation'] == server_generation) &
        (fleet_df['datacenter_id'].isin(LATENCY_TO_DC[latency])) &
        (~fleet_df['server_id'].isin(processed_servers))
    ]


def _get_sorted_dc_costs(dc_info: pd.DataFrame, latency: str) -> pd.DataFrame:
    """
    Get datacenters sorted by cost of energy (descending) for a given latency.
    """
    return dc_info[dc_info['datacenter_id'].isin(LATENCY_TO_DC[latency])].sort_values('cost_of_energy', ascending=False)


def _process_server_dismissals(dc_costs: pd.DataFrame, dismissable_servers: pd.DataFrame, 
                               excess: int, server_generation: str, server_slots: int,
                               actions: List[Dict], processed_servers: Set, 
                               datacenters_df: pd.DataFrame) -> Tuple[List[Dict], Set, pd.DataFrame]:
    """
    Process server dismissals starting from the most expensive datacenters.
    """
    for _, dc in dc_costs.iterrows():
        if excess == 0:
            break

        dc_servers = dismissable_servers[dismissable_servers['datacenter_id'] == dc['datacenter_id']]
        to_dismiss = min(len(dc_servers), excess)

        for _, server in dc_servers.head(to_dismiss).iterrows():
            actions.append({
                'datacenter_id': dc['datacenter_id'],
                'server_generation': server_generation,
                'server_id': server['server_id'],
                'action': 'dismiss'
            })

            processed_servers.add(server['server_id'])

        # Update datacenter capacity
        datacenters_df.loc[datacenters_df['datacenter_id'] == dc['datacenter_id'], 'used_slots'] -= to_dismiss * server_slots
        excess -= to_dismiss

    return actions, processed_servers, datacenters_df


def _allocate_servers(state: SystemState, demand: pd.DataFrame) -> List[Dict]:
    """
    Allocate servers based on the current system state and demand.

    This function performs a series of operations to optimize server allocation:
    1. Dismisses old servers
    2. Moves servers between datacenters
    3. Buys new servers
    4. Dismisses unused servers

    Args:
        state (SystemState): The current state of the system, including fleet and datacenter information.
        demand (pd.DataFrame): The current demand for servers.

    Returns:
        List[Dict]: A list of all actions taken (dismiss, move, buy) to optimize server allocation.
    """
    all_action = []

    # Dismiss old servers
    dismiss_old_srvs_actions, processed_servers = check_fleet_for_old_servers(state)
    state.update_fleet(dismiss_old_srvs_actions)
    state.update_datacenter_capacity()
    all_action += dismiss_old_srvs_actions
    
    srvs_cnt = srvs_count(state.fleet)
    demand_def_exc = calculate_demand_def_exc(srvs_cnt, demand)

    # Move servers
    move_actions, updated_demand_def_exc, processed_servers = move_servers(state, 
                                                                           demand_def_exc,
                                                                           processed_servers)
    state.update_fleet(move_actions)
    state.update_datacenter_capacity()
    all_action += move_actions

    # Buy servers
    buy_actions, updated_demand_def_exc, processed_servers = buy_servers(state, 
                                                                         updated_demand_def_exc, 
                                                                         processed_servers)
    state.update_fleet(buy_actions)
    state.update_datacenter_capacity()
    all_action += buy_actions

    # Dismiss unused servers
    dismiss_actions = dismiss_servers(state, 
                                      updated_demand_def_exc, 
                                      processed_servers)
    
    state.update_fleet(dismiss_actions)
    state.update_datacenter_capacity()
    all_action += dismiss_actions

    return all_action


def get_solution(actual_demand: pd.DataFrame, ma_window_size: int) -> List[Dict]:
    """
    Generate a solution for server allocation based on actual demand over time.

    This function simulates the server allocation process over a series of time steps,
    optimizing the allocation at each step based on the current demand.

    Args:
        actual_demand (pd.DataFrame): The actual demand for servers over time.

    Returns:
        List[Dict]: A list of all actions taken throughout the simulation to optimize server allocation.

    Note:
        - The function assumes a total of 168 time steps (e.g., hours in a week).
        - It uses a SystemState object to keep track of the current system state.
        - The tqdm library is used to display a progress bar during execution.
    """
    state = SystemState(DATACENTERS, SERVERS)

    ma_actual_demand = calculate_moving_average(actual_demand, window_size=ma_window_size)
    demand_srvs = calculate_servers(ma_actual_demand)
    
    for ts in tqdm(range(1, TOTAL_TIME_STEPS + 1)): 
        current_demand_srvs = demand_srvs.loc[ma_actual_demand['time_step'] == ts]
        actions = _allocate_servers(state, current_demand_srvs)
        
        state.update_time()
        state.update_solution(actions)
    
    return state.solution


def main():
    print(f'[TEST MODE]: Seed used: {TEST_SEED}; MA Window size: {MA_WINDOW_SIZE}')

    np.random.seed(TEST_SEED)

    actual_demand = get_actual_demand(DEMAND)
    solution = get_solution(actual_demand, ma_window_size=MA_WINDOW_SIZE)
    
    solution_df = pd.DataFrame(solution)
    solution_df.to_json(f'../data/solution_ma_w{MA_WINDOW_SIZE}.json', orient='records', indent=4)

if __name__ == "__main__":
    main()

