import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from evaluation_v6 import *
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
        self.demand = pd.DataFrame(self.csv_format_demand(demand))

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
        with open(json_file_path, 'r') as f:
            actions = json.load(f)
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

    def plot_demand(self, demand_df):
        server_types = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
        latency_sensitivities = ["low", "medium", "high"]
        
        fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 24))
        axes = axes.flatten()
        
        for i, server in enumerate(server_types):
            for j, latency in enumerate(latency_sensitivities):
                ax: plt.Axes = axes[i * len(latency_sensitivities) + j]
                subset = demand_df[demand_df["latency_sensitivity"] == latency]
                ax.plot(subset["time_step"], subset[server], label='Demand')
                
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
                ax.plot(range(1, len(capacity_data) + 1), capacity_data, label='Capacity')
            
                ax.set_title(f'Demand and Capacity for {server} ({latency} latency) Over Time')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
        
        plt.tight_layout()
        plt.show()

    def run(self):
        self.process_actions()
        self.plot_demand(self.demand)


seeds = known_seeds('test')
for seed in seeds: # type: ignore
    np.random.seed(seed)
    print(f"Seed: {seed}")
    actual_demand = get_actual_demand(pd.read_csv('./data/demand.csv')) 
    tracker = DemandTracker('greedy_profit_v2/output_test', f'{seed}.json', actual_demand)
    tracker.run()
    break
