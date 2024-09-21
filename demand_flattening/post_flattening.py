# Setup
import sys
import os

import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation import get_actual_demand
from utils import load_problem_data, save_json, load_json
from greedy_profit_v2.data import get_sorted_servers
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

json_file_path = './demand_flattening/1097_for_testing.json'
demand, datacenters, servers, selling_prices, elasticity = load_problem_data()
sorted_servers = get_sorted_servers('data/test_data/most_profitable_servers_by_artem.csv')

seed = 1097
np.random.seed(seed)
actual_demand = get_actual_demand(demand)

def csv_format_demand(demand):
    columns = ["time_step","latency_sensitivity","CPU.S1","CPU.S2","CPU.S3","CPU.S4"
    ,"GPU.S1","GPU.S2","GPU.S3"]
    d = {i:[] for i in columns}
    for k in ["low","medium","high"]:
        for timestep in range(1,demand["time_step"].max()+1):
            d["time_step"].append(timestep)
            d["latency_sensitivity"].append(k)
            gens_with_demand = []
            for rownum in range(len(demand[demand["time_step"] == timestep].index)):
                row = demand[demand["time_step"] == timestep].iloc[rownum]
                servergen = row["server_generation"]
                lat_d = row[k]
                d[servergen].append(lat_d)
                gens_with_demand.append(servergen)
            s = set(gens_with_demand)
            for i in ["CPU.S1","CPU.S2","CPU.S3","CPU.S4","GPU.S1","GPU.S2","GPU.S3"]:
                if i not in s:
                    d[i].append(0)
    return d


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
                for t in range(time_step, min(time_step + LIFE_EXPECTANCY, MAX_TIMESTEP + 1)):
                    current_server_capacity[dc_id][server_gen][t] += capacity
            elif action['action'] == 'dismiss':
                if server_id in servers:
                    buy_time = servers[server_id]['buy_time']
                    server_gen = servers[server_id]['server_generation']
                    for t in range(time_step, min(buy_time + LIFE_EXPECTANCY, MAX_TIMESTEP + 1)):
                        current_server_capacity[dc_id][server_gen][t] -= capacity
                    del servers[server_id]
    return current_server_capacity

overall_capacity_by_datacenter = process_actions(json_file_path)

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
results = []
for server_generation, latency_sensitivity in sorted_servers:
    # DEBUG
    # time_step = 156
    # server_generation = 'GPU.S3'
    # latency_sensitivity = 'medium'
    print(f'{server_generation} {latency_sensitivity}')

    # 1) Calculate the difference between the demand met and the actual demand for all time steps (delta D)
    formatted_demand = pd.DataFrame(csv_format_demand(actual_demand))
    relevant_demand = formatted_demand.query('latency_sensitivity == @latency_sensitivity')[server_generation].reset_index(drop=True)
    # print(relevant_demand.iloc[time_step - 1])
    relevant_capacity = overall_capacity_by_latency[latency_sensitivity][server_generation][1:]
    # print(relevant_capacity[time_step - 1])
    # demand delta is the amount of demand we want to change (negative = decrease, there is excess, positive vice versa)
    absolute_demand_delta = (relevant_capacity - relevant_demand).reset_index(drop=True)
    # print(absolute_demand_delta.iloc[time_step - 1])
    percentage_demand_delta = (absolute_demand_delta / relevant_demand).fillna(0).case_when(caselist=[(relevant_demand.eq(0), 0)])
    # print(percentage_demand_delta.iloc[time_step - 1])

    # 2) Given elasticity, calculate the delta p for all time steps
    def quick_query(df: pd.DataFrame, column: str):
        return df.query('server_generation == @server_generation and latency_sensitivity == @latency_sensitivity').get(column).iloc[0]
    this_elasticity = quick_query(elasticity, 'elasticity')
    price_delta = percentage_demand_delta / this_elasticity
    # print(price_delta.iloc[time_step - 1])

    # 3) Calculate the new price for all time steps
    base_price = quick_query(selling_prices, 'selling_price')
    # print(base_price)
    new_price: pd.Series = (price_delta * base_price) + base_price
    # print(new_price.iloc[time_step - 1])

    # print(f"{time_step} time step: deltaD={percentage_demand_delta.iloc[time_step - 1]} epsilon={this_elasticity} deltap={price_delta.iloc[time_step - 1]} basep={base_price} p={new_price.iloc[time_step - 1]}")

    # 4) Convert the new prices into price change actions with time step, latency sensitivity and server generation
    pricing_strategy = []
    for time_step, price in new_price.items():
        pricing_strategy.append({
            'time_step': time_step + 1,
            'latency_sensitivity': latency_sensitivity,
            'server_generation': server_generation,
            'price': price
        })

    # Trim the pricing strategy on either side until there is a change.
    
    # print(pricing_strategy[0])
    # print(len(pricing_strategy))

    #From the start
    no_change = True
    time_step = 1
    prev_price = base_price
    while no_change:
        current_price = {}
        for price_change in pricing_strategy:
            if price_change['time_step'] == time_step:
                current_price = price_change['price']
                break

        if prev_price == current_price:
            pricing_strategy.pop(0)
        else:
            no_change = False

        time_step += 1


    #From the end
    no_change = True
    time_step = 168
    for price_change in pricing_strategy:
        if price_change['time_step'] == time_step:
            prev_price = price_change['price']

    while no_change:
        time_step -= 1
        current_price = {}
        for price_change in pricing_strategy:
            if price_change['time_step'] == time_step:
                current_price = price_change['price']
                break

        if prev_price == current_price:
            pricing_strategy.pop()
        else:
            no_change = False


    print(pricing_strategy[0])
    print(pricing_strategy[-1])
    print(len(pricing_strategy))
    results.extend(pricing_strategy)

# print(results)

# 4) Save the pricing strategy in the JSON
fleet = load_json(json_file_path)["fleet"]
data = {
    "fleet": fleet,
    "pricing_strategy": results
}
save_json(f'./{seed}_demand_flattened_post.json', data)