import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from scipy.optimize import basinhopping
from max_profit2 import max_profit
from functools import partial
import json

class Big:
    def __init__(self):
        self.fleet = pd.DataFrame(columns=["time_step", "datacenter_id", "server_generation", "server_id"])
        self.server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.latencies = ["low","medium","high"]
        self.num_to_sgen = {i:self.server_generations[i] for i in range(7)}
        self.num_to_lat = {0:"low", 1:"medium", 2:"high"}
        self.server_id = 0
    
    def increment_id(self):
        self.server_id+=1

    def process_dataframe(self, result_df, step_size):
        buy_array = self.buy_all(result_df)
        discards = {}
        discard = []
        for i in range(24,168,24):
            discards[i] = result_df[result_df["time_step"] == i]
        for i in discards.keys():
            if(i<96):
                number_upto = np.arange(1,i,step_size)
            else:
                number_upto = np.arange(1+i-(96-step_size),i,step_size)
            for dc in range(4):
                # print(discards[i]["discards"].iloc[dc])
                local_discard_array = discards[i]["discards"].iloc[dc]
                datacenter_id = discards[i]["datacenter_id"].iloc[dc]
                for k in range(len(local_discard_array)):
                    if local_discard_array[k]>0:
                        #get how deep into array we are
                        finder = int(k/7)
                        #find the servergen timeslot/timestep that corresponds to
                        timeframe = number_upto[finder]
                        servergen = self.server_generations[k%7]
                        datacenter_id = datacenter_id
                        discard.append({"number":local_discard_array[k], "timestep":i ,"datacenter_id":datacenter_id, "server_generation":servergen, 
                        "discard_timestep":timeframe})
        dismiss_array = []
        for i in discard:
            dismiss_array.extend(self.dismiss(i["timestep"], i["discard_timestep"], i["datacenter_id"], i["server_generation"], i["number"]))
        buy_array.extend(dismiss_array)
        return buy_array

    def buy(self, timestep, datacenter_id, servergen, number):
        buy_array = []
        for i in range(number):
            buy_array.append({"time_step":timestep, "datacenter_id":datacenter_id, 
                "server_generation":servergen, "server_id":self.server_id, "action":"buy"})
            self.fleet.loc[len(self.fleet.index)] = [timestep, datacenter_id, servergen, self.server_id]
            self.increment_id()
        return buy_array

    def dismiss(self, timestep, servergen_timestep, datacenter_id, servergen, number):
        dismiss_array = []
        mask = (self.fleet["time_step"] == servergen_timestep) & (self.fleet["server_generation"] == servergen) & (self.fleet["datacenter_id"] == datacenter_id)
        owned_ids = self.fleet[mask]["server_id"]
        # print(number)
        # print(len(owned_ids.index))
        indexes_to_dismiss = []
        for i in range(number):
            dismiss_array.append({"time_step":int(timestep), "datacenter_id":datacenter_id, 
                "server_generation":servergen, "server_id":int(owned_ids.iloc[i]), "action":"dismiss"})
            indexes_to_dismiss.append(owned_ids.index[i])
            # print(timestep)
            # print(servergen)
            # print(servergen_timestep)
            # self.fleet = self.fleet.drop(owned_ids.index[0])
            # owned_ids = owned_ids.drop(owned_ids.index[0])
        self.fleet = self.fleet.drop(indexes_to_dismiss)
        return dismiss_array

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

    def generate_prices(self, rng, prices_step_size):
        selling_prices = pd.read_csv("../data/selling_prices.csv")
        lat_prices = []
        for i in range(3):
            sgen_prices = []
            for j in range(7):
                mask = (selling_prices["server_generation"] == self.num_to_sgen[j]) & (selling_prices["latency_sensitivity"] == self.num_to_lat[i])
                default_price = selling_prices[mask]["selling_price"].iloc[0]
                sgen_price = rng.uniform(default_price*1,default_price*1,int(168/prices_step_size))
                sgen_prices.append(sgen_price)
            lat_prices.append(np.array(sgen_prices))
        return np.array(lat_prices)
    
    def generate_bounds(self, step_size):
        selling_prices = pd.read_csv("../data/selling_prices.csv")
        lat_prices = []
        for i in range(3):
            sgen_prices = []
            for j in range(7):
                mask = (selling_prices["server_generation"] == self.num_to_sgen[j]) & (selling_prices["latency_sensitivity"] == self.num_to_lat[i])
                default_price = selling_prices[mask]["selling_price"].iloc[0]
                lower_bound = int(default_price*0.5)
                upper_bound = int(default_price*1.5)
                sgen_price = np.repeat([lower_bound, upper_bound],int(168/step_size))
                sgen_prices.append(sgen_price)
            lat_prices.append(np.array(sgen_prices))
        return np.array(lat_prices)
    
    def wrapper_function(self, x, demand, ls, prices_step_size, step_size, return_df, negative):
        return max_profit(prices=x, demand=demand, ls=ls, prices_step_size=prices_step_size, step_size=step_size, return_df=return_df, negative=negative)

    def simulated_annealing(self, n_iterations, demand, prices_step_size=24, step_size=12):
        rng = np.random.default_rng()
        bounds = self.generate_bounds(prices_step_size)
        very_best = []
        best = self.generate_prices(rng, prices_step_size)
        # best_profit = max_profit(demand, prices=best, prices_step_size=prices_step_size, step_size=step_size, return_df=False)
        for i in range(3):
            ls = [i]
            best_profit = max_profit(demand, ls=[i], prices=best[i], prices_step_size=prices_step_size, step_size=step_size, return_df=False)
            for k in range(n_iterations):
                curr = self.generate_prices(rng, prices_step_size)
                curr_profit = max_profit(demand, ls=[i], prices=curr[i], prices_step_size=prices_step_size, step_size=step_size, return_df=False)
                if(curr_profit>best_profit):
                    best_profit = curr_profit
                    best[i] = curr[i]
                # fixed_custom_function = partial(self.wrapper_function, demand = demand, ls=ls, prices_step_size=prices_step_size, step_size=step_size, return_df=False, negative=True)
                # ret = basinhopping(fixed_custom_function, best[i,j], niter=10, stepsize=5)
        return best

    def get_pricing_strat_df(self, prices, prices_step_size):
        pricing_strat = []
        for latency in range(3):
            for servergen in range(7):
                lat = self.num_to_lat[latency]
                sgen = self.num_to_sgen[servergen]
                for ts in range(int(168/prices_step_size)):
                    timestep = ts*prices_step_size
                    price = prices[latency,servergen,ts]
                    prices_dict = {"time_step":timestep, "latency_sensitivity":lat, "server_generation":sgen, "price":price}
                    pricing_strat.append(prices_dict)
        pricing_strategy = pd.DataFrame(data=pricing_strat)
        return pricing_strategy


    def csv_format_demand(self, demand):
        columns = ["time_step","latency_sensitivity","CPU.S1","CPU.S2","CPU.S3","CPU.S4"
        ,"GPU.S1","GPU.S2","GPU.S3"]
        dic = {i:[] for i in columns}
        for k in self.latencies:
            for timestep in range(1,demand["time_step"].max()+1):
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
