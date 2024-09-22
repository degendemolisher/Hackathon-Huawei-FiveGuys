import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import pandas as pd
from utils import load_json, save_json
from demand_flattening.post_flattening import post_flatten_demand
from greedy_profit_v2.greedy_profit_algorithm import greedy_profit_algorithm
from greedy_profit_v2.results import save_results_as_actions
from seeds import known_seeds
from evaluation import get_actual_demand

from demand_flattening.pre_flattening import pre_flatten_demand


def get_solution(enable_post_demand_flattening=False):
    
    seeds = known_seeds()

    demand = pd.read_csv('./data/demand.csv')
    for seed in seeds:
        print('----------------------------')
        print(f'Seed: {seed}')
        # SET THE RANDOM SEED
        np.random.seed(seed)

        # GET THE DEMAND
        actual_demand = get_actual_demand(demand)

        # CALL YOUR APPROACH HERE
        pricing_dict, new_demand = pre_flatten_demand(actual_demand)
        results = greedy_profit_algorithm(new_demand, pricing_dict, 0, float(8))

        # Extract the directory path from the file path
        file_path = f'output/solutions/{seed}.json'
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the results as actions
        save_results_as_actions(file_path, results)

        if enable_post_demand_flattening:
            # POST DEMAND FLATTENING
            pricing_strategy = post_flatten_demand(file_path, actual_demand)

            fleet = load_json(file_path)["fleet"]
            data = {
                "fleet": fleet,
                "pricing_strategy": pricing_strategy
            }
            save_json(file_path, data)

if __name__ == '__main__':
    get_solution(enable_post_demand_flattening=True)
    print('----------------------------')
    print('All seeds have been processed')