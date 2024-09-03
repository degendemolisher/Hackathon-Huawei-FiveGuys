import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
import json

class Big:
    def __init__(self):
        self.server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.latencies = ["low","medium","high"]
        self.server_id = 0
    
    def increment_id(self):
        self.server_id+=1

    def buy(self, timestep, datacenter_id, servergen, number):
        buy_array = []
        for i in range(number):
            buy_array.append({"time_step":timestep, "datacenter_id":datacenter_id, 
                "server_generation":servergen, "server_id":self.server_id, "action":"buy"})
            self.increment_id()
        return buy_array

    def buy_all(self, dataframe):
        buy_array = []
        for i in range(dataframe["time_step"].min(),dataframe["time_step"].max()+1):
            ts_datacenters_df = dataframe[dataframe["time_step"] == i]
            for j in range(len(ts_datacenters_df.index)):
                datacenter = ts_datacenters_df.iloc[j]["datacenter_id"]
                for servergen in self.server_generations:
                    number = int(ts_datacenters_df.iloc[j][servergen])
                    if(number == 0):
                        continue
                    bought = self.buy(i, datacenter, servergen, number)
                    buy_array.extend(bought)
        return buy_array

    def csv_format_demand(self, demand):
        columns = ["time_step","latency_sensitivity","CPU.S1","CPU.S2","CPU.S3","CPU.S4"
        ,"GPU.S1","GPU.S2","GPU.S3"]
        dic = {i:[] for i in columns}
        for k in self.latencies:
            for timestep in range(1,demand["time_step"].max()):
                dic["time_step"].append(timestep)
                dic["latency_sensitivity"].append(k)
                gens_with_demand = []
                #print(demand[demand["time_step"] == timestep])
                for rownum in range(len(demand[demand["time_step"] == timestep].index)):
                    row = demand[demand["time_step"] == timestep].iloc[rownum]
                    servergen = row["server_generation"]
                    lat_d = row[k]
                    dic[servergen].append(lat_d)
                    gens_with_demand.append(servergen)
                s = set(gens_with_demand)
                for i in self.server_generations:
                    if i not in s:
                        dic[i].append(0)
        return dic
