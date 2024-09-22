import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation import *
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
        self.demand_raw = demand
        self.demand = pd.DataFrame(self.csv_format_demand(demand))
        self.demand_new = {servergen: {latency: [] for latency in ["low", "medium", "high"]} for servergen in self.SERVER_GENERATIONS}

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
        elasticity = pd.read_csv('data/price_elasticity_of_demand.csv')
        base_prices = pd.read_csv('data/selling_prices.csv')
        with open(json_file_path, 'r') as f:
            # fleet
            file = json.load(f)
            actions = file['fleet']
            for action in actions:
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
                    for t in range(time_step, min(time_step + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP)):
                        self.current_server_capacity[dc_id][server_gen][t] += capacity
                elif action['action'] == 'dismiss':
                    if server_id in self.servers:
                        buy_time = self.servers[server_id]['buy_time']
                        server_gen = self.servers[server_id]['server_generation']
                        for t in range(time_step, min(buy_time + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP)):
                            self.current_server_capacity[dc_id][server_gen][t] -= capacity
                        del self.servers[server_id]

            # pricing
            pricing = file['pricing_strategy']
            pricing = pd.DataFrame(pricing)
            for latency in ["low", "medium", "high"]:
                for servergen in self.SERVER_GENERATIONS:
                    # find elasticity
                    elas_mask = (elasticity["server_generation"] == servergen) & (elasticity["latency_sensitivity"] == latency)
                    sgen_elasticity = elasticity[elas_mask]["elasticity"].iloc[0]
                    # subset of price change for each pair of servergen and latency
                    pricing_mask = (pricing["server_generation"] == servergen) & (pricing["latency_sensitivity"] == latency)
                    server_price = pricing[pricing_mask]
                    server_price = server_price.drop(columns=['latency_sensitivity', 'server_generation'])
                    # df with timesteps and empty price column
                    server_price_full = pd.DataFrame(columns=["time_step", "selling_price"])
                    server_price_full["time_step"] = range(1, self.MAX_TIMESTEP + 1)
                    merged_df = pd.merge(server_price_full, server_price, on='time_step', how='outer')
                    # add base price
                    base_price_mask = (base_prices["server_generation"] == servergen) & (base_prices["latency_sensitivity"] == latency)
                    base_price = base_prices[base_price_mask]["selling_price"].iloc[0]
                    timestep_0 = pd.DataFrame({'time_step': [0], 'price': [base_price]})
                    merged_df = merged_df.sort_values(by='time_step').reset_index(drop=True)
                    merged_df = pd.concat([timestep_0, merged_df], ignore_index=True)
                    # Copy values from 'y' to 'x' where 'y' is not NaN
                    merged_df['selling_price'] = merged_df['price'].combine_first(merged_df['selling_price'])
                    # Forward-fill the NaN values in 'x'
                    pd.set_option('future.no_silent_downcasting', True)
                    merged_df['selling_price'] = merged_df['selling_price'].ffill()
                    # get actual demand for servergen and latency
                    melted_demand = pd.melt(self.demand_raw, id_vars=['time_step', 'server_generation'], 
                                            value_vars=['high', 'low', 'medium'], 
                                            var_name='latency_sensitivity', 
                                            value_name='original_demand')
                    demand_mask = (melted_demand["server_generation"] == servergen) & (melted_demand["latency_sensitivity"] == latency)
                    new_demand = melted_demand[demand_mask]
                    # fill missing timesteps with demand 0
                    new_demand = pd.merge(merged_df[['time_step']], new_demand, on='time_step', how='left')
                    new_demand['original_demand'] = new_demand['original_demand'].fillna(0)
                    merged_df = pd.merge(merged_df, new_demand[['time_step', 'original_demand']], on='time_step', how='outer')
                    # dD = (P - P0) / P0 * e
                    # D = D0 * (1 + dD)
                    merged_df['new_demand'] = ((merged_df['selling_price'] - base_price) / base_price * sgen_elasticity + 1) * merged_df['original_demand']
                    # print(merged_df)
                    # put into dict for each servergen and latency
                    # take only the array of new_demand
                    self.demand_new[servergen][latency] = merged_df['new_demand'].values[1:]
                    # break
                # break
            # print(self.demand_new)
                    

    def plot_demand(self, demand_df):
        latency_sensitivities = ["low", "medium", "high"]
        
        fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 24))
        axes = axes.flatten()
        
        for i, server in enumerate(self.SERVER_GENERATIONS):
            for j, latency in enumerate(latency_sensitivities):
                ax: plt.Axes = axes[i * len(latency_sensitivities) + j]
                subset = demand_df[demand_df["latency_sensitivity"] == latency]
                # ax.plot(subset["time_step"], subset[server], label='Demand')
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
                ax.plot(range(len(capacity_data)), capacity_data, label='Capacity')

                # Extract and plot the new demand data
                new_demand_data = self.demand_new[server][latency]
                ax.plot(range(len(new_demand_data)), new_demand_data, label='New Demand')
            
                ax.set_title(f'Demand and Capacity for {server} ({latency} latency) Over Time')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
        
        plt.tight_layout()
        output_dir = './output/visuals/demand'
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
    tracker = DemandTracker('output/flattened/', f'{seed}.json', actual_demand)
    tracker.run()
    # break