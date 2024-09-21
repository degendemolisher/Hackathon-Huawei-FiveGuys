# Setup
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation import get_actual_demand
from utils import load_problem_data
from data import get_sorted_servers
import json
import numpy as np

DATACENTERS = ['DC1', 'DC2', 'DC3', 'DC4']
SERVER_GENERATIONS = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
SERVER_CAPACITY = {
        'CPU.S1': 60, 'CPU.S2': 75, 'CPU.S3': 120, 'CPU.S4': 160,
        'GPU.S1': 8, 'GPU.S2': 8, 'GPU.S3': 8
    }
MAX_TIMESTEP = 168
LIFE_EXPECTANCY = 96

demand, datacenters, servers, selling_prices, elasticity = load_problem_data()
sorted_servers = get_sorted_servers('data/test_data/most_profitable_servers_by_artem.csv')
actual_demand = get_actual_demand(demand)

seed = 1097
np.random.seed(seed)


# 2) Calculate the capacity at all time steps for each server generation and latency sensitivty
def process_actions(json_file_path):
    servers = {}
    current_server_capacity = {dc: {gen: [0] * (MAX_TIMESTEP + 1) for gen in SERVER_GENERATIONS} for dc in DATACENTERS}
    with open(json_file_path, 'r') as f:
        actions = json.load(f)['fleet']
        for action in actions:
            dc_id = action['datacenter_id']
            time_step = action['time_step']
            server_gen = action['server_generation']
            server_id = action['server_id']
            capacity = SERVER_CAPACITY[server_gen]

            if action['action'] == 'buy':
                servers[server_id] = {
                    'datacenter_id': dc_id,
                    'server_generation': server_gen,
                    'buy_time': time_step
                }
                for t in range(time_step, min(time_step + LIFE_EXPECTANCY, MAX_TIMESTEP)):
                    current_server_capacity[dc_id][server_gen][t] += capacity
            elif action['action'] == 'dismiss':
                if server_id in servers:
                    buy_time = servers[server_id]['buy_time']
                    server_gen = servers[server_id]['server_generation']
                    for t in range(time_step, min(buy_time + LIFE_EXPECTANCY, MAX_TIMESTEP)):
                        current_server_capacity[dc_id][server_gen][t] -= capacity
                    del servers[server_id]
    return current_server_capacity

seed = 1097
np.random.seed(seed)
overall_capacity_by_datacenter = process_actions('./demand_flattening/1097_for_testing.json')

overall_capacity_by_latency = {}
overall_capacity_by_latency['low'] = overall_capacity_by_datacenter['DC1']
del overall_capacity_by_datacenter['DC1']
overall_capacity_by_latency['medium'] = overall_capacity_by_datacenter['DC2']
del overall_capacity_by_datacenter['DC2']

overall_capacity_by_latency['high'] = {}
for server_generation in SERVER_GENERATIONS:
    dc3_data = np.array(overall_capacity_by_datacenter['DC3'][server_generation])
    dc4_data = np.array(overall_capacity_by_datacenter['DC4'][server_generation])
    merged_data = dc3_data + dc4_data
    overall_capacity_by_latency['high'][server_generation] = merged_data.tolist()

# 3) For each latency sensitivity/server generation
for server_generation, latency_sensitivity in sorted_servers:
    print(f'{server_generation} {latency_sensitivity}')

    # 1) Calculate the difference between the demand met and the actual demand for all time steps (delta D)
    relevant_demand = actual_demand.query(f'server_generation == @server_generation')[latency_sensitivity]
    relevant_capacity = overall_capacity_by_latency[latency_sensitivity][server_generation]
    demand_diff = relevant_demand - relevant_capacity
    print(relevant_demand)
    print(relevant_capacity)
    print(demand_diff)

    break
