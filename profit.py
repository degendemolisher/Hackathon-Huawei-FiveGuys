import pandas as pd
import numpy as np
import json

servers_df = pd.read_csv('data/servers.csv')
demand_df = pd.read_csv('data/demand.csv')
selling_prices_df = pd.read_csv('data/selling_prices.csv')
datacenters_df = pd.read_csv('data/datacenters.csv')
print("Loaded servers data:")
print(servers_df.head())

print("Loaded demand data:")
print(demand_df.head()) 

def parse_release_time(release_time_str):
    try:
        # Remove brackets and split by comma
        return tuple(map(int, release_time_str.strip('[]').split(',')))
    except Exception as e:
        print(f"Error parsing release time: {e}")
        return (0, 0)   
 
servers_df['release_time'] = servers_df['release_time'].apply(parse_release_time)
 
print("Parsed release times:")
print(servers_df[['release_time']].head())
 
time_steps = 168
actions = []
server_id_counter = 1   
 
server_lifespans = {}
datacenter_slots = {dc_id: 0 for dc_id in datacenters_df['datacenter_id']}


def calculate_profit(server_row, demand, selling_price, datacenter_energy_cost, failure_rate):
    capacity = server_row['capacity']
    adjusted_capacity = capacity * (1 - failure_rate)  # Adjusting capacity with failure rate
    met_demand = min(adjusted_capacity, demand)
    revenue = met_demand * selling_price
    
    energy_cost = server_row['energy_consumption'] * datacenter_energy_cost
    maintenance_cost = 1.5 * server_row['average_maintenance_fee']  # Simplified maintenance cost
    
    cost = server_row['purchase_price'] + energy_cost + maintenance_cost
    return revenue - cost
 
def is_server_available(release_time_range, time_step):
    start_time, end_time = release_time_range
    return start_time <= time_step <= end_time
 
for time_step in range(1, time_steps + 1):
    print(f"Processing Time Step: {time_step}")
     
    for _, datacenter in datacenters_df.iterrows():
        datacenter_id = datacenter['datacenter_id']
        energy_cost = datacenter['cost_of_energy']
        latency_sensitivity = datacenter['latency_sensitivity']
 
        available_servers = servers_df[servers_df['release_time'].apply(lambda x: is_server_available(x, time_step))]
        print(f"Available servers at time step {time_step}: {len(available_servers)}")
        
        # Debug: Print filtered servers
        if len(available_servers) > 0:
            print(available_servers.head())
        else:
            print(f"No servers available at time step {time_step}")
        
        best_profit = -float('inf')
        best_server = None
 
        for _, server_row in available_servers.iterrows():
            server_generation = server_row['server_generation']
            server_type = server_row['server_type']
 
            try:
                demand = demand_df.loc[
                    (demand_df['latency_sensitivity'] == latency_sensitivity),
                    f'{server_generation}'
                ].values[0]
                selling_price = selling_prices_df.loc[
                    (selling_prices_df['server_generation'] == server_generation) &
                    (selling_prices_df['latency_sensitivity'] == latency_sensitivity),
                    'selling_price'
                ].values[0]
                print(f"Demand: {demand}, Selling Price: {selling_price}")
            except IndexError:
                print(f"No matching demand or selling price for server generation {server_generation} at time step {time_step}")
                continue  # Skip if no matching demand or selling price
 
            failure_rate = np.random.uniform(0.05, 0.1)
 
            profit = calculate_profit(server_row, demand, selling_price, energy_cost, failure_rate)
            print(f"Profit for server {server_generation} at datacenter {datacenter_id}: {profit}")
 
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
                    "server_id": f"server_{server_id_counter}",  # Unique server ID
                    "action": "buy"
                }
                actions.append(action)
                server_id_counter += 1  # Increment server ID counter

                # Update datacenter slots usage and server lifespan
                datacenter_slots[datacenter_id] += slots_required
                server_lifespans[action['server_id']] = 0  # Start tracking lifespan

                # Print the action in the specified format
                print(json.dumps(action, indent=4))
            else:
                print(f"Datacenter {datacenter_id} does not have enough slots for {best_server['server_generation']}")
        else:
            print(f"No profitable server found for datacenter {datacenter_id} at time step {time_step}")
 
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
                "action": "dismiss"
            }
            actions.append(action)
            datacenter_slots[action['datacenter_id']] -= server_info['slots_size'] 
            del server_lifespans[server_id]  
            print(f"Server {server_id} dismissed at time step {time_step}")
 
print("\nFinal solution in required format:")
print(json.dumps(actions, indent=4))
 
with open('data/greedy_solution.json', 'w') as outfile:
    json.dump(actions, outfile, indent=4)

print("Greedy solution generated and printed successfully.")