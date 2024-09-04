import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from max_profit2 import *
from omerutils import Big

def get_my_solution(d):
    big = Big()
    #convert the garbage ass idiot demand format to csv format
    dd = big.csv_format_demand(d)
    ddd = pd.DataFrame.from_dict(dd)
    result_df = max_profit(ddd)
    valid = verify_solution_integrity(result_df)
    if(not valid):
        print("solution has a problem")
    # result_df2 = max_profit(ddd,115,35)
    buy_array = big.buy_all(result_df)
    #buy_array.extend(big.buy_all(result_df2))
    #buy_array = buy_array.flatten()
    return buy_array


seeds = known_seeds('training')

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')