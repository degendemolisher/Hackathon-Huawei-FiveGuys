import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation import *
from utils import load_problem_data, load_solution
from seeds import known_seeds


class DemandTracker():
    DATACENTERS = ['DC1', 'DC2', 'DC3', 'DC4']
    SERVER_GENERATIONS = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
    SERVER_CAPACITY = {
        'CPU.S1': 60, 'CPU.S2': 75, 'CPU.S3': 120, 'CPU.S4': 160,
        'GPU.S1': 8, 'GPU.S2': 8, 'GPU.S3': 8
    }
    MAX_TIMESTEP = 168
    LIFE_EXPECTANCY = 96

    def __init__(self, solution_json_dir, json_filename, demand):
        self.solution_json_dir = solution_json_dir
        self.json_filename = json_filename
        self.current_server_capacity = {dc: {gen: [0] * (self.MAX_TIMESTEP + 1) for gen in self.SERVER_GENERATIONS} for dc in self.DATACENTERS}
        self.servers = {}
        self.base_demand = demand
        self.demand = pd.DataFrame(self.csv_format_demand(demand))
        self.new_demand = pd.DataFrame()

    def csv_format_demand(self, demand):
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

    def process_actions(self):
        json_file_path = os.path.join(self.solution_json_dir, self.json_filename)
        fleet, pricing_strategy = load_solution(json_file_path)

        for _, action in fleet.iterrows():
            dc_id = action['datacenter_id']
            time_step = action['time_step']
            server_gen = action['server_generation']
            server_id = action['server_id']
            capacity = self.SERVER_CAPACITY[server_gen]

            if action['action'] == 'buy':
                self.servers[server_id] = {
                    'datacenter_id': dc_id,
                    'server_generation': server_gen,
                    'buy_time': time_step
                }
                for t in range(time_step, min(time_step + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP) + 1):
                    self.current_server_capacity[dc_id][server_gen][t] += capacity
            elif action['action'] == 'dismiss':
                if server_id in self.servers:
                    buy_time = self.servers[server_id]['buy_time']
                    server_gen = self.servers[server_id]['server_generation']
                    for t in range(time_step, min(buy_time + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP) + 1):
                        self.current_server_capacity[dc_id][server_gen][t] -= capacity
                    del self.servers[server_id]
                    
        # pricing
        # for each servergen and latency get mask
        base_prices = pd.read_csv('./data/selling_prices.csv')
        elasticity = pd.read_csv('./data/price_elasticity_of_demand.csv')
        # get base demand
        for servergen in self.SERVER_GENERATIONS:
            for latency in ['low', 'medium', 'high']:
                mask = (pricing_strategy['server_generation'] == servergen) & (pricing_strategy['latency_sensitivity'] == latency)
                server_new_prices = pricing_strategy[mask]
                print(server_new_prices)
                # make new price_change array (all NaN)
                price_change = pd.DataFrame()
                price_change['time_step'] = range(0, self.MAX_TIMESTEP + 1)
                price_change['price_multiplier'] = np.nan
                # fill in the new prices
                for _, row in server_new_prices.iterrows():
                    time_step = row['time_step']
                    new_price = row['price_multiplier']
                    price_change.loc[time_step, 'price_multiplier'] = new_price
                # fill in base price
                base_price = base_prices[(base_prices['server_generation'] == servergen) & (base_prices['latency_sensitivity'] == latency)]['selling_price'].values[0]
                price_change.loc[0, 'price_multiplier'] = base_price
                print(price_change)
                # divide the price column by the base price
                price_change['price'] = price_change['price'] / base_price
                print(price_change)
                # make a new column for the new demand
                price_change['new_demand'] = np.nan
                server_elasticity = elasticity[(elasticity['server_generation'] == servergen) & (elasticity['latency_sensitivity'] == latency)]['elasticity'].values[0]
                # fill in the new demand, demand = base_demand * (price_multiplier * elasticity)
                # if there is price change, fill in the new demand
                for i in range(1, self.MAX_TIMESTEP + 1):
                    if not np.isnan(price_change.loc[i, 'price_multiplier']):
                        price_multiplier = price_change.loc[i, 'price_multiplier']
                        price_change.loc[i, 'new_demand'] = self.base_demand.loc[i, servergen] * price_multiplier * server_elasticity

                break
            break

    def plot_demand(self, demand_df):
        latency_sensitivities = ["low", "medium", "high"]
        
        fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 24))
        axes = axes.flatten()
        
        for i, server in enumerate(self.SERVER_GENERATIONS):
            for j, latency in enumerate(latency_sensitivities):
                ax: plt.Axes = axes[i * len(latency_sensitivities) + j]
                subset = demand_df[demand_df["latency_sensitivity"] == latency]
                # Plot demand data
                ax.plot(subset["time_step"].values, subset[server].values, label='Demand')
                
                # Extract and plot the capacity data
                dc_to_latency = {
                    'low': ['DC1'],
                    'medium': ['DC2'],
                    'high': ['DC3', 'DC4']
                }
                capacity_data = []
                if latency == 'high':
                    # Merge the data of DC3 and DC4
                    dc3_data = self.current_server_capacity['DC3'][server]
                    dc4_data = self.current_server_capacity['DC4'][server]
                    merged_data = [dc3 + dc4 for dc3, dc4 in zip(dc3_data, dc4_data)]
                    capacity_data.extend(merged_data)
                else:
                    for dc in dc_to_latency[latency]:
                        capacity_data.extend(self.current_server_capacity[dc][server])
                # Plot capacity data
                ax.plot(range(len(capacity_data)), capacity_data, label='Capacity')
                # Plot new_demand data
                new_subset = self.new_demand[['time_step', 'server_generation', latency]]
                new_demand_data = new_subset[new_subset['server_generation'] == server]
                ax.plot(new_demand_data['time_step'], new_demand_data[latency], label=server, linestyle='dashed')
            
                ax.set_title(f'Demand and Capacity for {server} ({latency} latency) Over Time')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
        
        plt.tight_layout()
        output_dir = './output/visuals/demand/'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.json_filename))[0]
        output_file = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(output_file)
        plt.close()

    def run(self):
        self.process_actions()
        self.plot_demand(self.demand)


seeds = known_seeds()
for seed in seeds: # type: ignore
    np.random.seed(seed)
    print(f"Seed: {seed}")
    actual_demand = get_actual_demand(pd.read_csv('./data/demand.csv')) 
    tracker = DemandTracker('output/solutions/lp', f'{seed}.json', actual_demand)
    tracker.run()
    break
