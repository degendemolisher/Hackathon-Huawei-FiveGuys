import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from max_profit2 import *
from omerutils import Big

selling_prices = pd.read_csv("data/selling_prices.csv")
server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
latencies = ["low","medium","high"]
num_to_sgen = {i:server_generations[i] for i in range(7)}
num_to_lat = {0:"low", 1:"medium", 2:"high"}

def get_my_solution(d):
    elasticity = pd.read_csv("data/price_elasticity_of_demand.csv")
    most_profitable_list = pd.read_csv("data/most_profitable_servers_by_artem.csv")
    most_profitable_list = most_profitable_list[["boom", "server_generation", "latency_sensitivity"]]
    min_price_list = []
    min_dP_list = []
    for index, row in most_profitable_list.iterrows():
        sgen = row["server_generation"]
        latency = row["latency_sensitivity"]
        mask = (elasticity["server_generation"] == sgen) & (elasticity["latency_sensitivity"] == latency)
        sgen_elasticity = elasticity[mask]["elasticity"].iloc[0]
        min_dP = -1 * ((sgen_elasticity+1)/sgen_elasticity)
        min_dP/=2

        dc_selling_price = selling_prices[mask]["selling_price"].iloc[0]
        min_p = (min_dP + 1) * dc_selling_price
        min_price_list.append(min_p)
        min_dP_list.append(min_dP)
    most_profitable_list["min_dP"] = min_dP_list
    most_profitable_list["min_price"] = min_price_list
    most_profitable_list = most_profitable_list.sort_values(by="boom")
    prices = most_profitable_list["min_price"].values
    prices = np.reshape(np.array(prices), (3,7,1))
 
    big = Big()
    #convert the garbage ass idiot demand format to csv format
    dd = big.csv_format_demand(d)
    ddd = pd.DataFrame.from_dict(dd)

    prices_step_size = 168
    step_size = 6
    # prices = big.simulated_annealing(10, demand=ddd, prices_step_size=prices_step_size, step_size=12)

    rng = np.random.default_rng()
    prices = big.generate_prices(rng, prices_step_size)

    result_df, profit = max_profit(ddd, prices=prices, prices_step_size=prices_step_size, step_size=step_size)
    #result_df.to_csv('out2.csv', index=True)
    buy_array = big.process_dataframe(result_df, step_size=step_size)
    # print(buy_array)
    #buy_array.extend(big.buy_all(result_df2))
    #buy_array = buy_array.flatten()
    pricing_strategy = big.get_pricing_strat_df(prices, prices_step_size)
    # pricing_strategy = pd.DataFrame(data=[{
	# 					"time_step": 1,
	# 					"latency_sensitivity": "low",
	# 					"server_generation": "CPU.S1",
	# 					"price": 10
	# 					}],columns=["time_step"])
    return buy_array, pricing_strategy


seeds = known_seeds()

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution, pricing_strategy = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, pricing_strategy, f'./output/{seed}.json')