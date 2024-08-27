import pandas as pd
import numpy as np
import json
from seeds import known_seeds

# Load the data
servers_df = pd.read_csv('data/servers.csv')
demand_df = pd.read_csv('data/demand.csv')
selling_prices_df = pd.read_csv('data/selling_prices.csv')
datacenters_df = pd.read_csv('data/datacenters.csv')

# Helper functions
def get_known(attribute):
    if attribute == 'server_generation':
        return ['CPU.S1', 
                'CPU.S2', 
                'CPU.S3', 
                'CPU.S4', 
                'GPU.S1', 
                'GPU.S2', 
                'GPU.S3']
    elif attribute == 'latency_sensitivity':
        return ['high', 
                'medium', 
                'low']
    return []

def get_random_walk(n, min_val, max_val):
    return np.random.uniform(min_val, max_val, size=n)

def get_actual_demand(demand):
    actual_demand = []
    for ls in get_known('latency_sensitivity'):
        for sg in get_known('server_generation'):
            d = demand[demand['latency_sensitivity'] == ls]
            sg_demand = d[sg].values.astype(float)
            rw = get_random_walk(sg_demand.shape[0], 0, 2)
            sg_demand += (rw * sg_demand)

            ls_sg_demand = pd.DataFrame()
            ls_sg_demand['time_step'] = d['time_step']
            ls_sg_demand['server_generation'] = sg
            ls_sg_demand['latency_sensitivity'] = ls
            ls_sg_demand['demand'] = sg_demand.astype(int)
            actual_demand.append(ls_sg_demand)

    actual_demand = pd.concat(actual_demand, axis=0, ignore_index=True)
    actual_demand = actual_demand.pivot(index=['time_step', 'server_generation'], columns='latency_sensitivity')
    actual_demand.columns = actual_demand.columns.droplevel(0)
    actual_demand = actual_demand.loc[actual_demand[get_known('latency_sensitivity')].sum(axis=1) > 0]
    actual_demand = actual_demand.reset_index(['time_step', 'server_generation'], col_level=1, inplace=False)
    return actual_demand

# Parse release time function
def parse_release_time(release_time_str):
    try:
        return tuple(map(int, release_time_str.strip('[]').split(',')))
    except Exception as e:
        print(f"Error parsing release time: {e}")
        return (0, 0)  

servers_df['release_time'] = servers_df['release_time'].apply(parse_release_time)

# Function to calculate profit and server availability
def calculate_profit(server_row, demand, selling_price, datacenter_energy_cost, failure_rate):
    capacity = server_row['capacity']
    adjusted_capacity = capacity * (1 - failure_rate)
    met_demand = min(adjusted_capacity, demand)
    revenue = met_demand * selling_price

    energy_cost = server_row['energy_consumption'] * datacenter_energy_cost
    maintenance_cost = 1.5 * server_row['average_maintenance_fee']

    cost = server_row['purchase_price'] + energy_cost + maintenance_cost
    return revenue - cost

def is_server_available(release_time_range, time_step):
    start_time, end_time = release_time_range
    return start_time <= time_step <= end_time

 
seeds = known_seeds('training')

# Loop through each seed and generate a solution
for seed_value in seeds:
    np.random.seed(seed_value)  # Set the seed for reproducibility
    actions = []
    server_id_counter = 1  
    server_lifespans = {}
    datacenter_slots = {dc_id: 0 for dc_id in datacenters_df['datacenter_id']}

    actual_demand_df = get_actual_demand(demand_df)  # Calculate actual demand

    for time_step in range(1, 169):
        print(f"Processing Time Step: {time_step}")

        for _, datacenter in datacenters_df.iterrows():
            datacenter_id = datacenter['datacenter_id']
            energy_cost = datacenter['cost_of_energy']
            latency_sensitivity = datacenter['latency_sensitivity']

            available_servers = servers_df[servers_df['release_time'].apply(lambda x: is_server_available(x, time_step))]
            print(f"Available servers at time step {time_step}: {len(available_servers)}")

            best_profit = -float('inf')
            best_server = None

            for _, server_row in available_servers.iterrows():
                server_generation = server_row['server_generation']

                try:
                    demand = actual_demand_df.loc[
                        (actual_demand_df['time_step'] == time_step) &
                        (actual_demand_df['server_generation'] == server_generation),
                        latency_sensitivity
                    ].values[0]
                    selling_price = selling_prices_df.loc[
                        (selling_prices_df['server_generation'] == server_generation) &
                        (selling_prices_df['latency_sensitivity'] == latency_sensitivity),
                        'selling_price'
                    ].values[0]
                except IndexError:
                    continue

                failure_rate = np.random.uniform(0.05, 0.1)
                profit = calculate_profit(server_row, demand, selling_price, energy_cost, failure_rate)

                if profit > best_profit:
                    best_profit = profit
                    best_server = server_row

            if best_server is not None:
                slots_required = best_server['slots_size']
                if datacenter_slots[datacenter_id] + slots_required <= datacenter['slots_capacity']:
                    action = {
                        "time_step": time_step,
                        "datacenter_id": datacenter_id,
                        "server_generation": best_server['server_generation'],
                        "server_id": f"server_{server_id_counter}",
                        "action": "buy"
                    }
                    actions.append(action)
                    server_id_counter += 1  

                    datacenter_slots[datacenter_id] += slots_required
                    server_lifespans[action['server_id']] = 0  

        for server_id in list(server_lifespans.keys()):
            server_lifespans[server_id] += 1
            server_generation = next(item['server_generation'] for item in actions if item['server_id'] == server_id)
            server_info = servers_df[servers_df['server_generation'] == server_generation].iloc[0]
            if server_lifespans[server_id] >= server_info['life_expectancy']:
                action = {
                    "time_step": time_step,

                "datacenter_id": next(item['datacenter_id'] for item in actions if item['server_id'] == server_id),
                    "server_generation": server_generation,
                    "server_id": server_id,
                    "action": "buy"
                }
                actions.append(action)
                datacenter_slots[action['datacenter_id']] -= server_info['slots_size']
                del server_lifespans[server_id]
                print(f"Server {server_id} dismissed at time step {time_step}")

    # Save the solution to a JSON file
    solution_filename = f'data/{seed_value}.json'
    with open(solution_filename, 'w') as outfile:
        json.dump(actions, outfile, indent=4)

    print(f"Solution for seed {seed_value} saved to {solution_filename}")

print("All solutions generated and saved successfully.")