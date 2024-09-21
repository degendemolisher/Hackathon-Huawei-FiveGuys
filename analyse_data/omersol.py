import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from max_profit2 import *
from omerutils import Big

selling_prices = pd.read_csv("../data/selling_prices.csv")
num_to_sgen = {i:generations[i] for i in range(7)}
num_to_lat = {0:"low", 1:"medium", 2:"high"}

def get_my_solution(d):
    big = Big()
    #convert the garbage ass idiot demand format to csv format
    dd = big.csv_format_demand(d)
    ddd = pd.DataFrame.from_dict(dd)

    prices_step_size = 12
    step_size = 6
    prices = big.generate_prices(prices_step_size)

    result_df = max_profit(ddd, prices=prices, prices_step_size=prices_step_size, step_size=step_size)
    #result_df.to_csv('out2.csv', index=True)
    # valid = verify_solution_integrity(result_df)
    # if(not valid):
    #     print("solution has a problem")
    buy_array = big.process_dataframe(result_df, step_size=step_size)
    # print(buy_array)
    #buy_array.extend(big.buy_all(result_df2))
    #buy_array = buy_array.flatten()
    pricing_strategy = pd.DataFrame(data={
						"time_step": 1,
						"latency_sensitivity": "low",
						"server_generation": "CPU.S1",
						"price": 10
						},columns=["time_step"])
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